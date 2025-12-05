#include "gelu_ocl.h"

#include <CL/cl.h>
#include <stdexcept>
#include <string>
#include <cstring>

static inline void oclCheck(cl_int err, const char* msg)
{
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " (error " + std::to_string(err) + ")");
    }
}

static const char* geluKernelSrc = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;

    float x = input[gid];

    const float k0 = 0.7978845608028654f;   // sqrt(2.0/pi)
    const float k1 = 0.044715f;

    float x3 = x * x * x;
    float inner = k0 * (x + k1 * x3);
    float gelu = 0.5f * x * (1.0f + tanh(inner));

    output[gid] = gelu;
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    cl_int err;

    cl_uint numPlatforms = 0;
    oclCheck(clGetPlatformIDs(0, nullptr, &numPlatforms),
        "clGetPlatformIDs (count) failed");
    if (numPlatforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    oclCheck(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr),
        "clGetPlatformIDs (list) failed");

    if (platform < 0 || static_cast<cl_uint>(platform) >= numPlatforms) {
        throw std::runtime_error("Invalid platform index");
    }
    cl_platform_id chosenPlatform = platforms[platform];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(chosenPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        throw std::runtime_error("No GPU devices found on selected platform");
    }

    std::vector<cl_device_id> devices(numDevices);
    oclCheck(clGetDeviceIDs(chosenPlatform, CL_DEVICE_TYPE_GPU,
        numDevices, devices.data(), nullptr),
        "clGetDeviceIDs failed");

    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    oclCheck(err, "clCreateContext failed");

    cl_queue_properties props[] = { 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    oclCheck(err, "clCreateCommandQueue failed");

    const char* src = geluKernelSrc;
    size_t srcLen = std::strlen(geluKernelSrc);

    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err);
    oclCheck(err, "clCreateProgramWithSource failed");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            logSize, &log[0], nullptr);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("clBuildProgram failed:\n" + log);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    oclCheck(err, "clCreateKernel failed");

    const size_t bytes = n * sizeof(float);

    cl_mem inBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, const_cast<float*>(input.data()), &err);
    oclCheck(err, "clCreateBuffer input failed");

    cl_mem outBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        bytes, nullptr, &err);
    oclCheck(err, "clCreateBuffer output failed");

    int nInt = static_cast<int>(n);
    oclCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuf),
        "clSetKernelArg(0) failed");
    oclCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuf),
        "clSetKernelArg(1) failed");
    oclCheck(clSetKernelArg(kernel, 2, sizeof(int), &nInt),
        "clSetKernelArg(2) failed");

    size_t globalWorkSize[1] = { n };
    err = clEnqueueNDRangeKernel(queue, kernel, 1,
        nullptr, globalWorkSize, nullptr,
        0, nullptr, nullptr);
    oclCheck(err, "clEnqueueNDRangeKernel failed");

    oclCheck(clFinish(queue), "clFinish failed");

    std::vector<float> output(n);
    oclCheck(clEnqueueReadBuffer(queue, outBuf, CL_TRUE,
        0, bytes, output.data(),
        0, nullptr, nullptr),
        "clEnqueueReadBuffer failed");

    clReleaseMemObject(inBuf);
    clReleaseMemObject(outBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
