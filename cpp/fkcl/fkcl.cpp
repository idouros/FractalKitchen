// fkcl.cpp : Defines the entry point for the application.
//
#define CL_HPP_TARGET_OPENCL_VERSION  300

#include <filesystem>
#include <conio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "fkcl.h"
#include "kernels.h"
#include "helpers.h"

FractalParams DEFAULT_PARAMS;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << " Usage: " << argv[0] << " ConfigFile" << std::endl;
        return -1;
    }

    LOG_OUT("Reading config file...");
    ConfigParams configParams;
    boost::property_tree::ini_parser::read_ini(argv[1], configParams);
    FractalParams p;
    p.type = configParams.get("fractal.type", DEFAULT_PARAMS.type);
    p.max_iter = configParams.get<unsigned int>("fractal.max_iter", DEFAULT_PARAMS.max_iter);
    p.divergence_threshold = configParams.get<float>("fractal.divergence_threshold", DEFAULT_PARAMS.divergence_threshold);
    p.xtra_1 = configParams.get<float>("fractal.xtra_1", DEFAULT_PARAMS.xtra_1);
    p.xtra_1_label = configParams.get("fractal.xtra_1_label", DEFAULT_PARAMS.xtra_1_label);
    p.xtra_2 = configParams.get<float>("fractal.xtra_2", DEFAULT_PARAMS.xtra_2);
    p.xtra_2_label = configParams.get("fractal.xtra_2_label", DEFAULT_PARAMS.xtra_2_label);
    p.n_rows = configParams.get<size_t>("image.n_rows", DEFAULT_PARAMS.n_rows);
    p.n_cols = configParams.get<size_t>("image.n_cols", DEFAULT_PARAMS.n_cols);
    p.x_start = configParams.get<float>("image.x_start", DEFAULT_PARAMS.x_start);
    p.x_end = configParams.get<float>("image.x_end", DEFAULT_PARAMS.x_end);
    p.y_start = configParams.get<float>("image.y_start", DEFAULT_PARAMS.y_start);
    p.y_end = configParams.get<float>("image.y_end", DEFAULT_PARAMS.y_end);
    p.output_dir = configParams.get("image.output_dir", DEFAULT_PARAMS.output_dir);
    p.zoom_step = configParams.get<float>("image.zoom_step", DEFAULT_PARAMS.zoom_step);
    p.pan_step = configParams.get<int>("image.pan_step", DEFAULT_PARAMS.pan_step);
    p.platformId = configParams.get<size_t>("device.platformId", DEFAULT_PARAMS.platformId);
    p.deviceId = configParams.get<size_t>("device.deviceId", DEFAULT_PARAMS.deviceId);
    p.showDeviceList = configParams.get<bool>("device.showDeviceList", DEFAULT_PARAMS.showDeviceList);
 

    while(true)
    {
        LOG_OUT("Initializing essential parameters...");
        cl::ImageFormat format(CL_R, CL_FLOAT);
        auto pixel_step = -1.0f;
        if (!std::isnan(p.x_end) && std::isnan(p.y_end))
        {
            pixel_step = (p.x_end - p.x_start) / (float)p.n_cols;
            p.y_end = p.y_start + pixel_step * p.n_rows;
        }
        else if (!std::isnan(p.y_end) && std::isnan(p.x_end))
        {
            pixel_step = (p.y_end - p.y_start) / (float)p.n_rows;
            p.x_end = p.x_start + pixel_step * p.n_cols;
        }
        else
        {
            std::cout << "Invalid arguments! Exactly one of x_end, y_end must be defined. Exiting..." << std::endl;
            return -1;
        }

        if (p.showDeviceList)
        {
            LOG_OUT("Reading platforms and devices...");
            listDevices();
        }

        LOG_OUT("Getting the device...");
        cl::Device device = getDevice(p.platformId, p.deviceId);
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        LOG_OUT("Building the kernel...");
        std::string complexSource = readFile("complex.cl");
        std::string fullSource = complexSource + "\n" + kernels[p.type];
        cl::Program program(context, fullSource);
        EXEC_TIMED(cl_int err = buildKernel(program, device);)

        LOG_OUT("Running the kernel...");
        cl::Image2D image(context, CL_MEM_WRITE_ONLY, format, p.n_cols, p.n_rows);
        LOG_OUT("\t  Generating fractal of type: " + p.type);
        LOG_OUT("\tMaximum number of iterations: " + std::to_string(p.max_iter));
        LOG_OUT("\t        Divergence threshold: " + std::to_string(p.divergence_threshold));
        LOG_OUT("\t           Extra parameter 1: " + p.xtra_1_label + ": " + std::to_string(p.xtra_1));
        LOG_OUT("\t           Extra parameter 2: " + p.xtra_2_label + ": " + std::to_string(p.xtra_2));
        EXEC_TIMED(runKernel(program, image, queue, p.x_start, p.y_start, pixel_step, p.max_iter,
            p.divergence_threshold, p.xtra_1, p.xtra_2, p.n_cols, p.n_rows);)

        LOG_OUT("Generating fractal image...");
        std::vector<float> hostData = readBackImageData(p.n_cols, p.n_rows, queue, image);
        cv::Mat fractalImage = generateFractalImage(p.n_rows, p.n_cols, hostData);
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
        imshow("Display window", fractalImage);

        LOG_OUT("Image generation complete!");
        auto waitForKey = true;
        while (waitForKey)
        {
            auto key = cv::waitKeyEx(0);
            if (key >= 0 && key <= 255)
            {
                char keyChar = std::tolower(key);
                switch (keyChar)
                {
                case 'q':
                    LOG_OUT("Exiting...");
                    return 0;
                case 'h':
                    std::cout << "TODO : help text" << std::endl;
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
                default:
                    continue;
                }
            }
            else
            {
                // TODO: make this cross-platform (will work only on Windows)
                switch(key)
                {
                case 2490368: // Up arrow
                    panVertical(p.y_start, p.y_end, pixel_step, -p.pan_step);
                    waitForKey = false;
                    break;
                case 2621440: // Down arrow
                    panVertical(p.y_start, p.y_end, pixel_step, p.pan_step);
                    waitForKey = false;
                    break;
                case 2424832: // Left arrow
                    panHorizontal(p.x_start, p.x_end, p.y_end, pixel_step, -p.pan_step);
                    waitForKey = false;
                    break;
                case 2555904: // Right arrow
                    panHorizontal(p.x_start, p.x_end, p.y_end, pixel_step, p.pan_step);
                    waitForKey = false;
                    break;
                default:
                    continue;
                }
            }
        }
    }
}
