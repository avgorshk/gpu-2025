#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <mutex>

const char *kernel_src = R"(
__kernel void gelu(__global const float* input, __global float* output, const int n) {
    int id = get_global_id(0);
    if (id >= n) return;
    
    float x = input[id];
    float xxx = x * x * x;
    output[id] = 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * xxx)));

}
)";

static cl_context cont = nullptr;
static cl_command_queue que = nullptr;
static cl_program prog = nullptr;
static cl_device_id device_id = nullptr;
static std::once_flag once;

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
    size_t n = input.size();
    std::vector<float> output(n);

    if (input.empty())
        return output;

    int platform_id = platform;

    std::call_once(once, [platform_id]()
                   {
        cl_platform_id p[4];
        cl_uint np;
        clGetPlatformIDs(4, p, &np);
        int plat = (platform_id >= (int)np) ? 0 : platform_id;
        clGetDeviceIDs(p[plat], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        cont = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
        que = clCreateCommandQueue(cont, device_id, 0, NULL);
        prog = clCreateProgramWithSource(cont, 1, &kernel_src, NULL, NULL);
        clBuildProgram(prog, 1, &device_id, NULL, NULL, NULL); });

    size_t sz = n * sizeof(float);
    cl_mem in = clCreateBuffer(cont, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sz, (void *)input.data(), NULL);
    cl_mem out = clCreateBuffer(cont, CL_MEM_WRITE_ONLY, sz, NULL, NULL);

    cl_kernel k = clCreateKernel(prog, "gelu", NULL);

    clSetKernelArg(k, 0, sizeof(cl_mem), &in);
    clSetKernelArg(k, 1, sizeof(cl_mem), &out);
    int size = static_cast<int>(n);
    clSetKernelArg(k, 2, sizeof(int), &size);

    size_t global = n;
    clEnqueueNDRangeKernel(que, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(que, out, CL_TRUE, 0, sz, output.data(), 0, NULL, NULL);

    clReleaseMemObject(out);
    clReleaseMemObject(in);
    clReleaseKernel(k);

    return output;
}