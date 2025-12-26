#include "gelu_ocl.h"
#include <CL/cl.h>


const char* geluKernelSource = R"(
    __kernel void GeluKernel(__global const float* input, __global float* output, const int n) {
        int idx = get_global_id(0);
        if (idx < n) {
            float x = input[idx];
            output[idx] = 0.5f * x * (1.0f + tanh(0.797884f * (x + 0.044715f * (x * x * x))));
        }
    }
    )";


std::vector<float> GeluOCL(const std::vector<float>& inputVec) {
    int localSize = 256;
    const size_t n = inputVec.size(), memSize = n * sizeof(float);
    std::vector<float> outputVec(n);

    cl_platform_id clPlatform;
    cl_device_id clDevice;
    cl_context clCtx;
    cl_command_queue clQueue;
    cl_program clProg;
    cl_kernel clKernel;
    cl_mem bufInput, bufOutput;
    
    clGetPlatformIDs(1, &clPlatform, NULL);
    clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);

    clCtx = clCreateContext(NULL, 1, &clDevice, NULL, NULL, NULL);
    clQueue = clCreateCommandQueue(clCtx, clDevice, 0, NULL);
       
    clProg = clCreateProgramWithSource(clCtx, 1, &geluKernelSource, NULL, NULL);
    const char* buildOpts = "-cl-fast-relaxed-math";
    clBuildProgram(clProg, 1, &clDevice, buildOpts, NULL, NULL);
    clKernel = clCreateKernel(clProg, "GeluKernel", NULL);

    bufInput = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, memSize, NULL, NULL);
    bufOutput = clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, memSize, NULL, NULL);

    clEnqueueWriteBuffer(clQueue, bufInput, CL_TRUE, 0, memSize, inputVec.data(), 0, NULL, NULL);
    
    clSetKernelArg(clKernel, 0, sizeof(cl_mem), &bufInput);
    clSetKernelArg(clKernel, 1, sizeof(cl_mem), &bufOutput);
    clSetKernelArg(clKernel, 2, sizeof(int), &n);

    const size_t globalSize = (n + localSize - 1) / localSize * localSize;
    const size_t localWorkSize = localSize;  
    clEnqueueNDRangeKernel(clQueue, clKernel, 1, NULL, &globalSize, &localWorkSize, 0, NULL, NULL);
    clEnqueueReadBuffer(clQueue, bufOutput, CL_TRUE, 0, memSize, outputVec.data(), 0, NULL, NULL);

    clReleaseMemObject(bufInput);
    clReleaseMemObject(bufOutput);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProg);
    clReleaseCommandQueue(clQueue);
    clReleaseContext(clCtx);

    return outputVec;
}