#include "gelu_ocl.h"

static const char* kernel = R"CLC(
__kernel void gelu_kernel(__global const float* in,
                          __global float* out,
                          const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        const float x = in[i];
        const float c = 0.044715f;
        const float sqrt_2_over_pi = 0.7978845608028654f;

        float x3 = x * x * x;
        float z = sqrt_2_over_pi * (x + c * x3);
        float s = 1.0f / (1.0f + exp(-2.0f * z));
        out[i] = x * s;
    }
}
)CLC";

struct GeluOCLState {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_handle;
};

GeluOCLState* GeluOCL_Init(int platform_index) {
    GeluOCLState* s = new GeluOCLState();

    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> plats(numPlatforms);
    clGetPlatformIDs(numPlatforms, plats.data(), nullptr);
    s->platform = plats[platform_index];

    cl_uint numDevices = 0;
    clGetDeviceIDs(s->platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    std::vector<cl_device_id> devs(numDevices);
    clGetDeviceIDs(s->platform, CL_DEVICE_TYPE_GPU, numDevices, devs.data(), nullptr);
    s->device = devs[0];

    s->context = clCreateContext(nullptr, 1, &s->device, nullptr, nullptr, nullptr);
    s->queue = clCreateCommandQueue(s->context, s->device, 0, nullptr);

    const char* src = kernel;
    size_t srcSize = std::strlen(src);
    s->program = clCreateProgramWithSource(s->context, 1, &src, &srcSize, nullptr);
    clBuildProgram(s->program, 1, &s->device, nullptr, nullptr, nullptr);
    s->kernel_handle = clCreateKernel(s->program, "gelu_kernel", nullptr);

    return s;
}

void GeluOCL_Shutdown(GeluOCLState* s) {
    if (!s) { return; }
    clReleaseKernel(s->kernel_handle);
    clReleaseProgram(s->program);
    clReleaseCommandQueue(s->queue);
    clReleaseContext(s->context);
    delete s;
}

std::vector<float> GeluOCL_Run(GeluOCLState* s, const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    size_t bytes = n * sizeof(float);

    cl_mem inBuf = clCreateBuffer(s->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)input.data(), nullptr);
    cl_mem outBuf = clCreateBuffer(s->context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    int ni = static_cast<int>(n);
    clSetKernelArg(s->kernel_handle, 0, sizeof(cl_mem), &inBuf);
    clSetKernelArg(s->kernel_handle, 1, sizeof(cl_mem), &outBuf);
    clSetKernelArg(s->kernel_handle, 2, sizeof(int), &ni);

    const size_t local = 256;
    size_t global = ((n + local - 1) / local) * local;
    clEnqueueNDRangeKernel(s->queue, s->kernel_handle, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    clEnqueueReadBuffer(s->queue, outBuf, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(inBuf);
    clReleaseMemObject(outBuf);

    return output;
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform = 0) {
    static GeluOCLState* state = GeluOCL_Init(platform);
    return GeluOCL_Run(state, input);
}