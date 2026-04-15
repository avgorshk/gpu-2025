#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <mutex>

const char* KERNEL_SOURCE = R"(
__kernel void gelu(__global const float* inArr, __global float* outArr, const int N) {
    int gid = get_global_id(0);
    if (gid >= N) return;
    
    float val = inArr[gid];
    float val3 = val * val * val;
    outArr[gid] = 0.5f * val * (1.0f + tanh(0.7978845608028654f * (val + 0.044715f * val3)));
}
)";

static cl_context oclContext = nullptr;
static cl_command_queue oclQueue = nullptr;
static cl_program oclProgram = nullptr;
static cl_device_id oclDevice = nullptr;
static std::once_flag initFlag;

std::vector<float> GeluOCL(const std::vector<float>& inVec, int platIdx) {
    size_t vecSize = inVec.size();
    std::vector<float> outVec(vecSize);

    int chosenPlat = platIdx;

    std::call_once(initFlag, [chosenPlat]() {
        cl_platform_id platforms[4];
        cl_uint numPlatforms;
        clGetPlatformIDs(4, platforms, &numPlatforms);
        int selectedPlat = (chosenPlat >= (int)numPlatforms) ? 0 : chosenPlat;
        clGetDeviceIDs(platforms[selectedPlat], CL_DEVICE_TYPE_GPU, 1, &oclDevice, NULL);
        oclContext = clCreateContext(NULL, 1, &oclDevice, NULL, NULL, NULL);
        oclQueue = clCreateCommandQueue(oclContext, oclDevice, 0, NULL);
        oclProgram = clCreateProgramWithSource(oclContext, 1, &KERNEL_SOURCE, NULL, NULL);
        clBuildProgram(oclProgram, 1, &oclDevice, NULL, NULL, NULL);
    });

    size_t bytes = vecSize * sizeof(float);
    cl_mem memIn = clCreateBuffer(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, (void*)inVec.data(), NULL);
    cl_mem memOut = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    cl_kernel kernel = clCreateKernel(oclProgram, "gelu", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &memIn);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &memOut);
    int intN = static_cast<int>(vecSize);
    clSetKernelArg(kernel, 2, sizeof(int), &intN);

    size_t globalSize = vecSize;
    clEnqueueNDRangeKernel(oclQueue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(oclQueue, memOut, CL_TRUE, 0, bytes, outVec.data(), 0, NULL, NULL);

    clReleaseMemObject(memOut);
    clReleaseMemObject(memIn);
    clReleaseKernel(kernel);

    return outVec;
}
