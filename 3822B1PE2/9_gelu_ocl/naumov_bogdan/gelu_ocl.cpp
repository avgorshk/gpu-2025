#include "gelu_ocl.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const char* GELU_KERNEL = 
"#define M_SQRT1_2 0.7071067811865475f\n"
"__kernel void gelu_kernel(\n"
"    __global const float* input,\n"
"    __global float* output,\n"
"    const unsigned int n) {\n"
"    \n"
"    int idx = get_global_id(0);\n"
"    if (idx >= n) return;\n"
"    \n"
"    float x = input[idx];\n"
"    \n"
"    // Fast GELU approximation using erf (from paper: Gaussian Error Linear Units (GELUs))\n"
"    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))\n"
"    // Fast erf approximation using polynomial\n"
"    float z = x * M_SQRT1_2;\n"
"    float t = 1.0f / (1.0f + 0.3275911f * fabs(z));\n"
"    \n"
"    // Polynomial approximation for erf\n"
"    float erf_approx = 1.0f - (0.254829592f * t - 0.284496736f * t * t + \n"
"                              1.421413741f * t * t * t - 1.453152027f * t * t * t * t + \n"
"                              1.061405429f * t * t * t * t * t) * exp(-z * z);\n"
"    \n"
"    if (z < 0.0f) erf_approx = -erf_approx;\n"
"    \n"
"    output[idx] = 0.5f * x * (1.0f + erf_approx);\n"
"}\n";

struct OpenCLCache {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    bool initialized;
    
    OpenCLCache() : platform(0), device(0), context(0), queue(0), 
                   program(0), kernel(0), initialized(false) {}
    
    ~OpenCLCache() {
        Cleanup();
    }
    
    void Cleanup() {
        if (kernel) {
            clReleaseKernel(kernel);
            kernel = 0;
        }
        if (program) {
            clReleaseProgram(program);
            program = 0;
        }
        if (queue) {
            clReleaseCommandQueue(queue);
            queue = 0;
        }
        if (context) {
            clReleaseContext(context);
            context = 0;
        }
        initialized = false;
    }
};

static OpenCLCache cache;

bool InitOpenCL(int platform_idx) {
    if (cache.initialized) return true;
    
    cl_int err;
    
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        return false;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platforms" << std::endl;
        return false;
    }
    
    if (platform_idx < 0 || platform_idx >= (int)num_platforms) {
        std::cerr << "Platform index out of range, using platform 0" << std::endl;
        platform_idx = 0;
    }
    
    cache.platform = platforms[platform_idx];

    err = clGetDeviceIDs(cache.platform, CL_DEVICE_TYPE_GPU, 1, &cache.device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "No GPU device found, trying CPU" << std::endl;
        err = clGetDeviceIDs(cache.platform, CL_DEVICE_TYPE_CPU, 1, &cache.device, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "No OpenCL devices found" << std::endl;
            return false;
        }
    }

    char device_name[128];
    clGetDeviceInfo(cache.device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::cout << "Using device: " << device_name << std::endl;

    cache.context = clCreateContext(NULL, 1, &cache.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context: " << err << std::endl;
        return false;
    }

    cache.queue = clCreateCommandQueue(cache.context, cache.device, 
                                      CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << err << std::endl;
        return false;
    }

    const char* kernel_source = GELU_KERNEL;
    
    cache.program = clCreateProgramWithSource(cache.context, 1, 
                                             &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program: " << err << std::endl;
        return false;
    }

    err = clBuildProgram(cache.program, 1, &cache.device, 
                        "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program: " << err << std::endl;

        size_t log_size;
        clGetProgramBuildInfo(cache.program, cache.device, 
                             CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(cache.program, cache.device, 
                             CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        
        return false;
    }

    cache.kernel = clCreateKernel(cache.program, "gelu_kernel", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel: " << err << std::endl;
        return false;
    }
    
    cache.initialized = true;
    return true;
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {

    if (!InitOpenCL(platform)) {
        throw std::runtime_error("Failed to initialize OpenCL");
    }
    
    size_t n = input.size();
    std::vector<float> output(n);
    
    if (n == 0) {
        return output;
    }
    
    cl_int err;

    cl_mem input_buffer = clCreateBuffer(cache.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        n * sizeof(float), (void*)input.data(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create input buffer: " + std::to_string(err));
    }
    
    cl_mem output_buffer = clCreateBuffer(cache.context, CL_MEM_WRITE_ONLY,
                                         n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        throw std::runtime_error("Failed to create output buffer: " + std::to_string(err));
    }

    err = clSetKernelArg(cache.kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(cache.kernel, 1, sizeof(cl_mem), &output_buffer);
    err |= clSetKernelArg(cache.kernel, 2, sizeof(unsigned int), &n);
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        throw std::runtime_error("Failed to set kernel arguments: " + std::to_string(err));
    }

    size_t global_size = n;
    size_t local_size = 256;

    if (global_size % local_size != 0) {
        global_size = ((global_size + local_size - 1) / local_size) * local_size;
    }
    
    cl_event kernel_event;
    err = clEnqueueNDRangeKernel(cache.queue, cache.kernel, 1, NULL,
                                &global_size, &local_size, 0, NULL, &kernel_event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        throw std::runtime_error("Failed to execute kernel: " + std::to_string(err));
    }

    err = clEnqueueReadBuffer(cache.queue, output_buffer, CL_TRUE, 0,
                             n * sizeof(float), output.data(), 1, &kernel_event, NULL);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        throw std::runtime_error("Failed to read results: " + std::to_string(err));
    }

    clFinish(cache.queue);
    
    clReleaseEvent(kernel_event);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    
    return output;
}