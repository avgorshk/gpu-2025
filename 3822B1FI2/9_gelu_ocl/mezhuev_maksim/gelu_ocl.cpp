#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <cstring>

static const char* kGeluKernelSource = R"CLC(
#define ROOT_TWO_OVER_PI 0.7978845608028654f
#define CUBIC_COEFF 0.044715f

inline float my_tanh_approx(float v) {
    float e2 = exp(2.0f * v);
    return (e2 - 1.0f) / (e2 + 1.0f);
}

__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = in[i];
    float x3 = x * x * x;
    float t = ROOT_TWO_OVER_PI * (x + CUBIC_COEFF * x3);
    float y = 0.5f * x * (1.0f + my_tanh_approx(t));
    out[i] = y;
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input, int platformIndex) {
    std::vector<float> output(input.size());
    size_t n = input.size();
    if (n == 0) return output;

    cl_int err = CL_SUCCESS;

    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static cl_program program = nullptr;
    static cl_kernel kernel = nullptr;
    static cl_mem inputBuffer = nullptr;
    static cl_mem outputBuffer = nullptr;
    static size_t allocatedElements = 0;
    static int currentPlatform = -1;

    if (!context || platformIndex != currentPlatform) {
        if (kernel) { clReleaseKernel(kernel); kernel = nullptr; }
        if (program) { clReleaseProgram(program); program = nullptr; }
        if (queue) { clReleaseCommandQueue(queue); queue = nullptr; }
        if (context) { clReleaseContext(context); context = nullptr; }
        if (inputBuffer) { clReleaseMemObject(inputBuffer); inputBuffer = nullptr; }
        if (outputBuffer) { clReleaseMemObject(outputBuffer); outputBuffer = nullptr; }
        allocatedElements = 0;

        cl_uint numPlatforms = 0;
        if (clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS || numPlatforms == 0)
            return output;

        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        if (platformIndex < 0 || platformIndex >= static_cast<int>(numPlatforms))
            return output;

        cl_platform_id platform = platforms[platformIndex];

        cl_uint numDevices = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) != CL_SUCCESS || numDevices == 0)
            return output;

        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        cl_device_id device = devices[0];

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (!context || err != CL_SUCCESS)
            return output;

    #if defined(CL_VERSION_2_0)
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    #else
        queue = clCreateCommandQueue(context, device, 0, &err);
    #endif
        if (!queue || err != CL_SUCCESS)
            return output;

        const char* src = kGeluKernelSource;
        size_t srcLen = std::strlen(src);
        program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err);
        if (!program || err != CL_SUCCESS)
            return output;

        err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            clReleaseProgram(program);
            program = nullptr;
            return output;
        }

        kernel = clCreateKernel(program, "gelu_kernel", &err);
        if (!kernel || err != CL_SUCCESS)
            return output;

        currentPlatform = platformIndex;
    }

    if (!inputBuffer || allocatedElements < n) {
        if (inputBuffer) clReleaseMemObject(inputBuffer);
        if (outputBuffer) clReleaseMemObject(outputBuffer);

        inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) return output;

        outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) return output;

        allocatedElements = n;
    }

    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_FALSE, 0, n * sizeof(float),
                               input.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return output;

    int n_int = static_cast<int>(n);
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n_int);
    if (err != CL_SUCCESS) return output;

    size_t localSize = 256;
    size_t globalSize = ((n + localSize - 1) / localSize) * localSize;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return output;

    clFinish(queue);

    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, n * sizeof(float),
                              output.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return output;

    return output;
}
