#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <CL/opencl.hpp>

// --- Miscellaneous helpers

void sectionInfo(const std::string& s)
{
    std::cout << "===================================================" << std::endl << s << std::endl;
}

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
    const float x_step,
    const float y_step,
    const size_t n_cols, 
    const size_t n_rows
)
{
    // TODO: Add timing information
    cl::Kernel kernel(program, "init_image");
    kernel.setArg(0, image);
    kernel.setArg(1, x_start);
    kernel.setArg(2, y_start);
    kernel.setArg(3, x_step);
    kernel.setArg(4, y_step);
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
    // TODO: Make it more colorful...
    cv::Mat fractalImage((int)n_rows, (int)n_cols, CV_32FC3);
    for (auto j = 0; j < n_cols; j++)
    {
        for (auto i = 0; i < n_rows; i++)
        {
            auto val = hostData[i * n_rows + j];
            fractalImage.at<cv::Vec3f>(i, j) = { val, val, val };
        } 
    }
    return fractalImage;
}
