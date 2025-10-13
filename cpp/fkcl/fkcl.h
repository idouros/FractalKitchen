#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <CL/opencl.hpp>


// --- OpenCL device management

void listDevices()
{
    std::cout << std::endl << "========== Reading platforms and devices ==========" << std::endl;

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
    std::cout << std::endl << "===================================================" << std::endl;
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
// Utility to read a file into a string
std::string readFile(const std::string& filename) {
    std::string sourceDir = std::filesystem::path(__FILE__).parent_path().string();
    std::string fullPath = sourceDir + "\\" + filename;
    std::ifstream file(fullPath);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + fullPath);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}