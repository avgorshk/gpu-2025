#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <mutex>

const char *CODE = R"(
__kernel void gelu(__global const float* input, __global float* output, const int size) {
    int id = get_global_id(0);
    if (id >= size) return;
    
    float x = input[id];
    float xxx = x * x * x;
    output[id] = 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * xxx)));
}
)";

static cl_program prog = nullptr;
static cl_device_id devic_e = nullptr;

static cl_context contex_t = nullptr;
static cl_command_queue queu_e = nullptr;

static std::once_flag once;

std::vector<float> GeluOCL(const std::vector<float> &input, int platform)
{
    size_t size = input.size();
    std::vector<float> output(size);

    int platform_id = platform;

    std::call_once(once, [platform_id]()
                   {
        cl_platform_id p[4];
        cl_uint np;
        clGetPlatformIDs(4, p, &np);
        int plat = (platform_id >= (int)np) ? 0 : platform_id;
        clGetDeviceIDs(p[plat], CL_DEVICE_TYPE_GPU, 1, &devic_e, NULL);
        contex_t = clCreateContext(NULL, 1, &devic_e, NULL, NULL, NULL);
        queu_e = clCreateCommandQueue(contex_t, devic_e, 0, NULL);
        prog = clCreateProgramWithSource(contex_t, 1, &CODE, NULL, NULL);
        clBuildProgram(prog, 1, &devic_e, NULL, NULL, NULL); });

    size_t sz = size * sizeof(float);
    cl_mem in = clCreateBuffer(contex_t, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sz, (void *)input.data(), NULL);
    cl_mem out = clCreateBuffer(contex_t, CL_MEM_WRITE_ONLY, sz, NULL, NULL);
    cl_kernel k = clCreateKernel(prog, "gelu", NULL);
    clSetKernelArg(k, 0, sizeof(cl_mem), &in);
    clSetKernelArg(k, 1, sizeof(cl_mem), &out);
    int int_size = static_cast<int>(size);

    clSetKernelArg(k, 2, sizeof(int), &int_size);
    size_t g = size;
    clEnqueueNDRangeKernel(queu_e, k, 1, NULL, &g, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queu_e, out, CL_TRUE, 0, sz, output.data(), 0, NULL, NULL);
    clReleaseMemObject(out);
    clReleaseMemObject(in);

    clReleaseKernel(k);

    return output;
}