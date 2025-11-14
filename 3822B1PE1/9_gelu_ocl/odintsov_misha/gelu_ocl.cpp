#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

const std::string gelu_kernel_code = R"(
    __kernel void gelu(__global float* input, __global float* output, const unsigned int size) {
        int i = get_global_id(0);
        if (i < size) {
            float x = input[i];
            float result = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
            output[i] = result;
        }
    }
)";



std::vector<float> GeluOCL(const std::vector<float>& input, int platform_id) {
    cl_int err;

  
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
 

    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
  
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    

    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);


    size_t local_work_size = std::min(max_work_group_size, input.size());
    size_t global_work_size = (input.size() + local_work_size - 1) / local_work_size * local_work_size;


    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), (void*)input.data(), &err);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float), nullptr, &err);
  
    const char* kernel_source = gelu_kernel_code.c_str();
    size_t source_size = gelu_kernel_code.size();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, &source_size, &err);
   

    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    
    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
   
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    

    unsigned int size = input.size();
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &size);
   
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    
    std::vector<float> output(input.size());
    clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, input.size() * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}