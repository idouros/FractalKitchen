// fkcl.cpp : Defines the entry point for the application.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "fkcl.h"

const char* kernelSource = R"CLC(
__kernel void init_image(write_only image2d_t img) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // float4 is required for image writes, only .x is used
    float4 pixel = (float4)(x, 0.0f, 0.0f, 0.0f);
    write_imagef(img, (int2)(x, y), pixel);
}
)CLC";


int main()
{
	// TODO: Make these configurable
    size_t n_rows = 400;
    size_t n_cols = 400;
	size_t platformId = 1;
	size_t deviceId = 0;

	// Get the device
	//listDevices();
	cl::Device device = getDevice(platformId, deviceId);
	cl::Context context(device);
    cl::CommandQueue queue(context, device);

	// Create an empty image matrix on the device
	cl::ImageFormat format(CL_R, CL_FLOAT);
	cl::Image2D image(context, CL_MEM_WRITE_ONLY, format, n_cols, n_rows);

    // Build and run the kernel
    cl::Program program(context, kernelSource);
    program.build({ device });
    cl::Kernel kernel(program, "init_image");
    kernel.setArg(0, image);
    cl::NDRange global(n_cols, n_rows);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();

    // Read back image data
    std::vector<float> hostData(n_cols * n_rows);
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { n_cols, n_rows, 1 };
    size_t r1 = 0;
    size_t r2 = 0;
    cl_int err = clEnqueueReadImage(queue(), image(), CL_TRUE, origin, region, 0, 0, hostData.data(), 0, nullptr, nullptr);

    // Generate the coloured fractal image
	cv::Mat fractalImage((int)n_rows, (int)n_cols, CV_8UC3); 
    for (size_t j = 0; j < n_cols; j++)
    {
        for (size_t i = 0; i < n_rows; i++)
        {
            auto val = (unsigned int) hostData[i * n_rows + j];
            fractalImage.at<cv::Vec3b>(i, j) = { 0, (uchar)val, 0 };
        }
    }

    // Display the image and wait for key press to close
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	imshow("Display window", fractalImage);
	cv::waitKey(0);


	return 0;
}
