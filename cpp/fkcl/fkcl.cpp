// fkcl.cpp : Defines the entry point for the application.
//
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "fkcl.h"

// TODO: kernels in separate file(s) / folder(s)
const char* kernelSource = R"CLC(
__kernel void init_image(write_only image2d_t img, float x_start, float y_start, float x_step, float y_step) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float y = y_start + (float)col * y_step;
    float x = x_start + (float)row * x_step;

    float ret = 0.0f;
    unsigned int max_iter = 10000;
    unsigned int i = 0;
    bool keep_going = true;
    cfloat z = (cfloat)(0.0f, 0.0f);
    cfloat c = (cfloat)(x, y);
    while(keep_going)
    {
        z = c_add(c_mul(z, z), c);
        i += 1;
        if( (i >= max_iter) || (c_abs(z) > 2) )
        {
            keep_going = false;
        }
    }
    if(c_abs(z) > 2)
    {
        ret = 1.0f;
    }

    // float4 is required for image writes
    float4 pixel = (float4)(ret, 0.0f, 0.0f, 0.0f);
    write_imagef(img, (int2)(row, col), pixel);
}
)CLC";


int main()
{
	// TODO: Make these configurable
    size_t n_rows = 500;
    size_t n_cols = 500;
	size_t platformId = 0;
	size_t deviceId = 0;
    float x_start = -2.5;
    float x_end = 1.5;
    float y_start = -2.0;
    float y_end = 2.0;

    auto x_step = (x_end - x_start) / (float)n_cols;
    auto y_step = (y_end - y_start) / (float)n_rows;

	// Get the device
	//listDevices();
	cl::Device device = getDevice(platformId, deviceId);
	cl::Context context(device);
    cl::CommandQueue queue(context, device);

	// Create an empty image matrix on the device
	cl::ImageFormat format(CL_R, CL_FLOAT);
	cl::Image2D image(context, CL_MEM_WRITE_ONLY, format, n_cols, n_rows);

    // Build the kernel
    std::string complexSource = readFile("complex.cl");
    std::string fullSource = complexSource + "\n" + kernelSource;
    cl::Program program(context, fullSource);
    cl_int err = program.build({ device }, "-cl-fast-relaxed-math");
    cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
    if (status != CL_BUILD_SUCCESS) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build failed:\n" << log << std::endl;
        return -1;
    }
    else {
        std::cout << "Build successful!" << std::endl;
    }

    // Run the kernel
    cl::Kernel kernel(program, "init_image");
    kernel.setArg(0, image);
    kernel.setArg(1, x_start);
    kernel.setArg(2, y_start);
    kernel.setArg(3, x_step);
    kernel.setArg(4, y_step);
    cl::NDRange global(n_cols, n_rows);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();

    // Read back image data
    std::vector<float> hostData(n_cols * n_rows);
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { n_cols, n_rows, 1 };
    size_t r1 = 0;
    size_t r2 = 0;
    err = clEnqueueReadImage(queue(), image(), CL_TRUE, origin, region, 0, 0, hostData.data(), 0, nullptr, nullptr);

    // Generate the coloured fractal image
	cv::Mat fractalImage((int)n_rows, (int)n_cols, CV_32FC3);
    for (size_t j = 0; j < n_cols; j++)
    {
        for (size_t i = 0; i < n_rows; i++)
        {
            auto val = hostData[i * n_rows + j];
            fractalImage.at<cv::Vec3f>(i, j) = { val, val, val };
        }
    }

    // Display the image and wait for key press to close
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	imshow("Display window", fractalImage);
	cv::waitKey(0);
	return 0;
}
