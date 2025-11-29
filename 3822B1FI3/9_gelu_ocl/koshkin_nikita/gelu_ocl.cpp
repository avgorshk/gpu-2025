#include "gelu_ocl.h"

#define BLOCK_SIZE 256

const char* GeluKernel = R"(
    __kernel void GeluKernel(__global const float* in, __global float* out, const int size) {
        int idx = get_global_id(0);
        if (idx < size) {
            float x = in[idx];
            out[idx] = 0.5f * x * (1.0f + tanh(0.797884f * (x + 0.044715f * (x * x * x))));
        }
    }
    )";


std::vector<float> GeluOCL(const std::vector<float>& input) {

    const size_t size = input.size(), memory = size * sizeof(float);
    std::vector<float> result(size);

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

    const size_t grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    const size_t block = BLOCK_SIZE;  
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &grid, &block, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, out, CL_TRUE, 0, memory, result.data(), 0, NULL, NULL);


    clReleaseMemObject(in);
    clReleaseMemObject(out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}