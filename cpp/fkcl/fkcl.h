/*********************************************************
 * fkcl.h — Application specific helpers and funtions
 * Author: Yannis Douros
 *********************************************************/
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <CL/opencl.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "helpers.h"
#include "colouring.h"

typedef boost::property_tree::ptree ConfigParams;

struct FractalParams {
    std::string type = "Mandelbrot";
    unsigned int max_iter = 100;
    double divergence_threshold = 2.0;
    double xtra_1 = -0.0;
    std::string xtra_1_label = "?";
    double xtra_2 = 0.0;
    std::string xtra_2_label = "?";
    size_t n_cols = 1200;
    size_t n_rows = 900;
    double x_start = -0.02;
    double x_end = 0.02;
    double y_start = -0.015;
    double y_end = std::numeric_limits<double>::quiet_NaN();
    std::string output_dir = "/fractals";
    double zoom_step = 0.05;
    int pan_step = 10;
    bool showDeviceList = false;
    size_t platformId = 0;
    size_t deviceId = 0;
};


// --- OpenCL device management

void listDevices()
{
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    cl_platform_id* platform_ids = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platform_ids, &num_platforms);

    for (size_t j = 0; j < num_platforms; j++)
    {
        std::cout << "------------------" << std::endl;
        cl_platform_id platform_id = platform_ids[j];
        size_t platform_info_length = 0;
        clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, nullptr, &platform_info_length);
        char buf[100]{};
        clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_info_length, buf, &platform_info_length);
        std::cout << "Platform " << j << ": " << buf << std::endl;
        cl_uint num_devices = 0;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        cl_device_id* device_ids = new cl_device_id[num_devices];
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids, &num_devices);

        for (size_t i = 0; i < num_devices; i++)
        {
            cl_device_id device_id = device_ids[i];
            size_t device_info_length = 0;
            clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &device_info_length);
            char buf[100]{};
            clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_info_length, buf, &device_info_length);
            std::cout << "Device " << j << ":" << i << ": " << buf << std::endl;
        }
        delete[] device_ids;
    }

    // clean up
    delete[] platform_ids;
}

cl::Device getDevice(const size_t& platformId, const size_t& deviceId)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[platformId];
    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
    cl::Device device = devices[deviceId];
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    return device;
}

// --- Kernel utilities

// Read a (kernel code) file into a string
std::string readFile(const std::string& filename) {
    std::string sourceDir = std::filesystem::path(__FILE__).parent_path().string();
    std::string fullPath = sourceDir + "\\" + filename;
    std::ifstream file(fullPath);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + fullPath);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// Build a kernel program
cl_int buildKernel(cl::Program& program, cl::Device& device)
{
    cl_int err = program.build({ device }, "-cl-fast-relaxed-math");
    cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
    if (status != CL_BUILD_SUCCESS) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build failed:\n" << log << std::endl;
    }
    else {
        std::cout << "Build successful!" << std::endl;
    }
    return err;
}

// --- Run a kernel
void runKernel(
    cl::Program& program,
    cl::Image2D& image,
    cl::CommandQueue& queue,
    const double x_start,
    const double y_start,
    const double pixel_step,
    const unsigned int max_iter,
    const double divergence_threshold,
    const double xtra_1,
    const double xtra_2,
    const size_t n_cols, 
    const size_t n_rows
)
{
    cl::Kernel kernel(program, "init_image");
    kernel.setArg(0, image);
    kernel.setArg(1, x_start);
    kernel.setArg(2, y_start);
    kernel.setArg(3, pixel_step);
    kernel.setArg(4, max_iter);
    kernel.setArg(5, divergence_threshold);
    kernel.setArg(6, xtra_1);
    kernel.setArg(7, xtra_2);
    cl::NDRange global(n_cols, n_rows);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();
}

// Read back image from the device to the host
std::vector<float> readBackImageData(const size_t n_cols, const size_t n_rows, cl::CommandQueue& queue, const cl::Image2D& image)
{
    std::vector<float> hostData(n_cols * n_rows);
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { n_cols, n_rows, 1 };
    size_t r1 = 0;
    size_t r2 = 0;
    cl_int err = clEnqueueReadImage(queue(), image(), CL_TRUE, origin, region, 0, 0, hostData.data(), 0, nullptr, nullptr);
    return hostData;
}

// Image Generation
cv::Mat generateFractalImage(const size_t n_rows, const size_t n_cols, const std::vector<float>& hostData, 
    const ColourMode& colourMode = COLOUR_MODE_BBCW)
{
    const float colourCycles = 1.0f; // TODO: make this user-configurable
    cv::Mat fractalImageBGR((int)n_rows, (int)n_cols, CV_8UC3);
    cv::Mat fractalImageHSV;

    if(colourMode == COLOUR_MODE_HSV)
    {
        fractalImageHSV = cv::Mat((int)n_rows, (int)n_cols, CV_8UC3);
    }

    for (auto j = 0; j < n_cols; j++)
    {
        for (auto i = 0; i < n_rows; i++)
        {
            auto val = hostData[i * n_cols + j];
            switch(colourMode)
            {
                case COLOUR_MODE_HSV:
                {
                    fractalImageHSV.at<cv::Vec3b>(i, j) = smoothHSV(val, colourCycles);
                    break;
                }
                case COLOUR_MODE_BBCW:
                    fractalImageBGR.at<cv::Vec3b>(i, j) = smoothBBCW(val, colourCycles);
                    break;
                case COLOUR_MODE_FLAME:
                    fractalImageBGR.at<cv::Vec3b>(i, j) = smoothFlame(val, colourCycles);
                    break;
                default:
                    LOG_OUT("Invalid Colour Mode! Exiting...")
                    exit(-1);
            }
        } 
    }

    if(colourMode == COLOUR_MODE_HSV)
    {
        cv::cvtColor(fractalImageHSV, fractalImageBGR, cv::COLOR_HSV2BGR);
    }
    return fractalImageBGR;
}

inline std::string formatTimestamp(std::time_t t) 
{
    auto now = std::chrono::system_clock::now();
    return std::format("{:%Y%m%d_%H%M%S}", now);
}

// Image navigation

#define NAVIGATION_ERROR    -1
#define NAVIGATION_END      0
#define NAVIGATION_RECALC   1
#define NAVIGATION_REDRAW   2

#define IMAGE_WINDOW "Fractal Kitchen"

#ifdef _WIN32
#define ARROW_KEY_UP    2490368
#define ARROW_KEY_DOWN  2621440    
#define ARROW_KEY_LEFT  2424832    
#define ARROW_KEY_RIGHT 2555904   
#elif __linux__
#define ARROW_KEY_UP    65362
#define ARROW_KEY_DOWN  65364  
#define ARROW_KEY_LEFT  65361  
#define ARROW_KEY_RIGHT 65363  
#endif

void saveFractalImageAndConfig(const cv::Mat& fractalImage, const FractalParams& p)
{
    if (!std::filesystem::exists(p.output_dir))
    {
        std::filesystem::create_directory(p.output_dir);
    }

    // Save the image

    auto output_file_name = "fk_" + p.type + "_" + formatTimestamp(std::time(nullptr));
    std::filesystem::path image_file_path = std::filesystem::path(p.output_dir) / (output_file_name + ".png");
    cv::imwrite(image_file_path.string(), fractalImage);
    LOG_OUT("Saved: " + image_file_path.string());

    // Save the config

    std::filesystem::path config_file_path = std::filesystem::path(p.output_dir) / (output_file_name + ".config");
    ConfigParams configParams;
    
    boost::property_tree::ptree p_fractal;
    p_fractal.add("type", p.type);
    p_fractal.add("max_iter", p.max_iter);
    p_fractal.add("divergence_threshold", p.divergence_threshold);
    p_fractal.add("xtra_1", p.xtra_1);
    p_fractal.add("xtra_1_label", p.xtra_1_label);
    p_fractal.add("xtra_2", p.xtra_2);
    p_fractal.add("xtra_2_label", p.xtra_2_label);
    configParams.add_child("fractal", p_fractal);

    boost::property_tree::ptree p_image;
    p_image.add("n_rows", p.n_rows);
    p_image.add("n_cols", p.n_cols);
    p_image.add("x_start", p.x_start);
    p_image.add("x_end", p.x_end);
    p_image.add("y_start", p.y_start);
    p_image.add("output_dir", p.output_dir);
    p_image.add("zoom_step", p.zoom_step);
    p_image.add("pan_step", p.pan_step);
    configParams.add_child("image", p_image);

    boost::property_tree::ptree p_device;
    p_device.add("platformId", p.platformId);
    p_device.add("deviceId", p.deviceId);
    p_device.add("showDeviceList", p.showDeviceList);
    configParams.add_child("device", p_device);

    boost::property_tree::ini_parser::write_ini(config_file_path.string(), configParams);
    LOG_OUT("Saved: " + config_file_path.string());
}

inline void zoom(double& x_start, double& x_end, double& y_start, double& y_end, const double zoom)
{
    auto x_mid = (x_start + x_end) / 2.0;
    auto y_mid = (y_start + y_end) / 2.0;

    auto x_offset = x_mid - x_start;
    auto y_offset = y_mid - y_start;

    x_start = x_mid - x_offset * (1 - zoom);
    x_end = x_mid + x_offset * (1 - zoom);
    y_start = y_mid - y_offset * (1 - zoom);
    y_end = std::numeric_limits<double>::quiet_NaN();
}

inline void panVertical(double& y_start, double& y_end, const double pixel_step, const int n_pixels)
{
    y_start -= pixel_step * n_pixels;
    y_end = std::numeric_limits<double>::quiet_NaN();
}

inline void panHorizontal(double& x_start, double& x_end, double& y_end, const double pixel_step, const int n_pixels)
{
    x_start -= pixel_step * n_pixels;
    x_end -= pixel_step * n_pixels;
    y_end = std::numeric_limits<double>::quiet_NaN();
}

inline void adjustMaxIter(FractalParams& p, const bool& upwards = true)
{
    const auto f = upwards ? 1.0 : -1.0;
    auto before = p.max_iter;
    auto after = (unsigned int)(p.max_iter * (1.0 + f * p.zoom_step / 2.0));
    if (after == before) 
    {
        p.max_iter++;
    }
    else
    {
        p.max_iter = after;
    }
    p.y_end = std::numeric_limits<double>::quiet_NaN();
}

inline void adjustDivergenceThreshold(FractalParams& p, const bool& upwards = true)
{
    const auto f = upwards ? 1.0 : -1.0;
    auto before = p.divergence_threshold;
    auto after = p.divergence_threshold + (f * 0.05);
    if (after >= 0.0)
    {
        p.divergence_threshold = after;
    }
    p.y_end = std::numeric_limits<double>::quiet_NaN();
}

inline void showHelpText()
{
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "\t+\t\tZoom in" << std::endl;
    std::cout << "\t-\t\tZoom out" << std::endl;
    std::cout << "\t<\t\tDecrease maximum iterations limit" << std::endl;
    std::cout << "\t>\t\tIncrease maximum iterations limit" << std::endl;
    std::cout << "\t[\t\tDecrease divergence threshold" << std::endl;
    std::cout << "\t]\t\tIncrease divergence threshold" << std::endl;
    std::cout << "\t(arrows)\tPan the image up/down - left/right" << std::endl;
    std::cout << "\tS\t\tSave the current image and the config parameters" << std::endl;
    std::cout << "\tH\t\tDisplay this help text" << std::endl;
    std::cout << "\tQ\t\tQuit" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
}

void cycleHue(cv::Mat& fractalImage, const int& step = 1)
{
    for (int j = 0; j < fractalImage.cols; j++)
    {
        for (int i = 0; i < fractalImage.rows; i++)
        {
            fractalImage.at<cv::Vec3b>(i, j)[0] = (fractalImage.at<cv::Vec3b>(i, j)[0] + step) % 255;
        }
    }
}

int navigateFractalImage(cv::Mat& fractalImage, FractalParams& p, double& pixel_step)
{
    auto waitForKey = true;
    while (waitForKey)
    {
        double prop = cv::getWindowProperty(IMAGE_WINDOW, cv::WND_PROP_VISIBLE);
        if (prop < 1) {
            LOG_OUT("Window closed by User. Exiting...");
            return NAVIGATION_END;
        }

        LOG_OUT("Press a key (H for help) > ");
        auto key = cv::waitKeyEx(0);
        if (key >= 0 && key <= 255)
        {
            char keyChar = std::tolower(key);
            switch (keyChar)
            {
            case 'q':
                LOG_OUT("Exiting...");
                return NAVIGATION_END;
            case 'h':
                showHelpText();
                break;
            case 's':
                LOG_OUT("Saving image and config...");
                saveFractalImageAndConfig(fractalImage, p);
                break;
            case '+':
                zoom(p.x_start, p.x_end, p.y_start, p.y_end, p.zoom_step);
                waitForKey = false;
                break;
            case '-':
                zoom(p.x_start, p.x_end, p.y_start, p.y_end, -p.zoom_step);
                waitForKey = false;
                break;
            case '<':
            case ',':
                adjustMaxIter(p, false);
                waitForKey = false;
                break;
            case '>':
            case '.':
                adjustMaxIter(p); 
                waitForKey = false;
                break;
            case '[':
            case '{':
                adjustDivergenceThreshold(p, false);
                waitForKey = false;
                break;
            case ']':
            case '}':
                adjustDivergenceThreshold(p);
                waitForKey = false;
                break;    
            case 'c':
                cycleHue(fractalImage, 8);
                return NAVIGATION_REDRAW;
            default:
                continue;
            }
        }
        else
        {
            switch (key)
            {
            case ARROW_KEY_UP:
                panVertical(p.y_start, p.y_end, pixel_step, -p.pan_step);
                waitForKey = false;
                break;
            case ARROW_KEY_DOWN:
                panVertical(p.y_start, p.y_end, pixel_step, p.pan_step);
                waitForKey = false;
                break;
            case ARROW_KEY_LEFT:
                panHorizontal(p.x_start, p.x_end, p.y_end, pixel_step, -p.pan_step);
                waitForKey = false;
                break;
            case ARROW_KEY_RIGHT:
                panHorizontal(p.x_start, p.x_end, p.y_end, pixel_step, p.pan_step);
                waitForKey = false;
                break;
            default:
                continue;
            }
        }
    }
    return NAVIGATION_RECALC;
}

