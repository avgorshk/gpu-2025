#include "gelu_ocl.h"
#include <CL/cl.h>

using std::vector;
const int block_size = 16

const char* GeluKernel = R"(
    __kernel void GeluKernel(__global const float* in, __global float* out, const int size) {
        int index = get_global_id(0);
        if (index < size) {
            float var = in[index];
            float cube = var * var * var;
            out[index] = 0.5f * x * (1.0f + tanh(0.797884f * (x + 0.044715f * cube)));
        }
    }
    )";

vector<float> GeluOCL(const vector<float>& input) {

    int size = input.size();
    int memory = size * sizeof(float);
    std::vector<float> ans(size);

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem in, out;
    
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
       
    program = clCreateProgramWithSource(context, 1, &GeluKernel, NULL, NULL);
    const char* args = "-cl-fast-relaxed-math";
    clBuildProgram(program, 1, &device, args, NULL, NULL);
    kernel = clCreateKernel(program, "GeluKernel", NULL);

    in = clCreateBuffer(context, CL_MEM_READ_ONLY, memory, NULL, NULL);
    out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, memory , NULL, NULL);

    clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, memory, input.data(), 0, NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    int grid = (size + block_size - 1) / block_size * block_size;
    int block = block_size;  
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &grid, &block, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, out, CL_TRUE, 0, memory, ans.data(), 0, NULL, NULL);

    clReleaseMemObject(in);
    clReleaseMemObject(out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return ans;
}
