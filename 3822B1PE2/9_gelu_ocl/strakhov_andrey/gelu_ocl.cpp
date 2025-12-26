#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <mutex>

const char *CODE = R"(
__kernel void kernel(__global const float* input, __global float* output, const int size) {
    int id = get_global_id(0);
    float M_PI = 3.14159265358979323846f;
    float calcCoef = sqrt(2.0f / M_PI);
    if (id < size) {
    
    float x = input[id];
     output[id] = 0.5f * x * (1.0f + tanh(calcCoef * (x + 0.044715f * x * x * x)));
}
)";

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_program prog = nullptr;
static cl_device_id device = nullptr;
static std::once_flag once;

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
    size_t size = input.size();
    std::vector<float> output(size);

    std::call_once(once, [platform]()
                   {
        cl_platform_id p[4];
        cl_uint np;
        clGetPlatformIDs(4, p, &np);
        int calc_platform = (platform >= (int)np) ? 0 : platform;
        clGetDeviceIDs(p[calc_platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device, 0, NULL);
        prog = clCreateProgramWithSource(context, 1, &CODE, NULL, NULL);
        clBuildProgram(prog, 1, &device, NULL, NULL, NULL); });

    int bitsize = size * sizeof(float);
    cl_mem in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bitsize, (void *)input.data(), NULL);
    cl_mem out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bitsize, NULL, NULL);

    cl_kernel kernel = clCreateKernel(prog, "kernel", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, out, CL_TRUE, 0, bitsize, output.data(), 0, NULL, NULL);

    clReleaseMemObject(out);
    clReleaseMemObject(in);
    clReleaseKernel(kernel);

    return output;
}