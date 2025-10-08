// fkcl.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <iostream>
#include <CL/cl.h>

// TODO: Reference additional headers your program requires here.

void listDevices()
{
    std::cout << std::endl << "========== PART 1: Reading platforms and devices ==========" << std::endl;

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