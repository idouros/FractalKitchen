#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <CL/opencl.hpp>
#include <opencv2/imgproc.hpp>
#include "helpers.h"

struct FractalParams {
    std::string type = "Mandelbrot";
    unsigned int max_iter = 100;
    float divergence_threshold = 2.0f;
    float xtra_1 = -0.0f;
    std::string xtra_1_label = "?";
    float xtra_2 = 0.0f;
    std::string xtra_2_label = "?";
    size_t n_cols = 1200;
    size_t n_rows = 900;
    float x_start = -0.02f;
    float x_end = 0.02f;
    float y_start = -0.015f;
    float y_end = std::numeric_limits<float>::quiet_NaN();
    std::string output_dir = "/fractals";
    float zoom_step = 0.05f;
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
    const float x_start,
    const float y_start,
    const float pixel_step,
    const unsigned int max_iter,
    const float divergence_threshold,
    const float xtra_1,
    const float xtra_2,
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

// --- Image Generation
cv::Mat generateFractalImage(const size_t n_rows, const size_t n_cols, const std::vector<float>& hostData)
{
    cv::Mat fractalImageHSV((int)n_rows, (int)n_cols, CV_8UC3);
    for (auto j = 0; j < n_cols; j++)
    {
        for (auto i = 0; i < n_rows; i++)
        {
            auto val = hostData[i * n_cols + j];
            auto h = static_cast<int>(std::lround(val * 179.0f)) + 180;
            auto s = 127;
            auto v = isAlmostEqual(val, 0.0f) ? 0 : 255;
            fractalImageHSV.at<cv::Vec3b>(i, j) = cv::Vec3b(h, s, v);
        } 
    }
    cv::Mat fractalImageBGR;
    cv::cvtColor(fractalImageHSV, fractalImageBGR, cv::COLOR_HSV2BGR);
    return fractalImageBGR;
}

std::string formatTimestamp(std::time_t t) 
{
    auto now = std::chrono::system_clock::now();
    return std::format("{:%Y%m%d_%H%M%S}", now);
}

void saveFractalImageAndConfig(const cv::Mat& fractalImage, const FractalParams& p)
{
    if (!std::filesystem::exists(p.output_dir))
    {
        std::filesystem::create_directory(p.output_dir);
    }
    auto output_file_name = "fk_" + p.type + "_" + formatTimestamp(std::time(nullptr));
    std::filesystem::path image_file_path = std::filesystem::path(p.output_dir) / (output_file_name + ".png");
    cv::imwrite(image_file_path.string(), fractalImage);
    LOG_OUT("Saved: " + image_file_path.string());
    // TODO : also save the config
    //std::filesystem::path config_file_path = std::filesystem::path(p.output_dir) / (output_file_name + ".config");
}

void zoom(float& x_start, float& x_end, float& y_start, float& y_end, const float zoom)
{
    auto x_mid = (x_start + x_end) / 2.0f;
    auto y_mid = (y_start + y_end) / 2.0f;

    auto x_offset = x_mid - x_start;
    auto y_offset = y_mid - y_start;

    x_start = x_mid - x_offset * (1 - zoom);
    x_end = x_mid + x_offset * (1 - zoom);
    y_start = y_mid - y_offset * (1 - zoom);
    y_end = std::numeric_limits<float>::quiet_NaN();
}

void panVertical(float& y_start, float& y_end, const float pixel_step, const int n_pixels)
{
    y_start -= pixel_step * n_pixels;
    y_end = std::numeric_limits<float>::quiet_NaN();
}

void panHorizontal(float& x_start, float& x_end, float& y_end, const float pixel_step, const int n_pixels)
{
    x_start -= pixel_step * n_pixels;
    x_end -= pixel_step * n_pixels;
    y_end = std::numeric_limits<float>::quiet_NaN();
}
