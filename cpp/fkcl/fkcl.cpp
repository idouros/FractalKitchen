// fkcl.cpp : Defines the entry point for the application.
//
#define CL_HPP_TARGET_OPENCL_VERSION  300

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "fkcl.h"
#include "kernels.h"

int main()
{
	// TODO: Make these configurable
    size_t n_cols = 800;
    size_t n_rows = 600;
	size_t platformId = 1;
	size_t deviceId = 0;
    float x_start = -2.5;
    float x_end = 1.5;
    float y_start = -1.5;
    float y_end = 1.5;
    bool showDeviceList = false;

    sectionInfo("Initializing essential parameters...");
    cl::ImageFormat format(CL_R, CL_FLOAT);
    auto x_step = (x_end - x_start) / (float)n_cols;
    auto y_step = (y_end - y_start) / (float)n_rows;

    if (showDeviceList)
    {
        sectionInfo("Reading platforms and devices...");
        listDevices();
    }

    sectionInfo("Getting the device...");
    cl::Device device = getDevice(platformId, deviceId);
	cl::Context context(device);
    cl::CommandQueue queue(context, device);

    sectionInfo("Building the kernel...");
    std::string complexSource = readFile("complex.cl");
    std::string fullSource = complexSource + "\n" + kernelMandelbrot;
    cl::Program program(context, fullSource);
    cl_int err = buildKernel(program, device);

    sectionInfo("Running the kernel...");
    cl::Image2D image(context, CL_MEM_WRITE_ONLY, format, n_cols, n_rows);
    runKernel(program, image, queue, x_start, y_start, x_step, y_step, n_cols, n_rows);

    sectionInfo("Generating fractal image...");
    std::vector<float> hostData = readBackImageData(n_cols, n_rows, queue, image);
    cv::Mat fractalImage = generateFractalImage(n_rows, n_cols, hostData);
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	imshow("Display window", fractalImage);

    sectionInfo("Image generation complete! Press any key to exit...");
    // TODO: Add save option
	cv::waitKey(0);
	return 0;
}
