#include "gelu_ocl.h"
#include <CL/opencl.hpp>

static const char* KERNEL_SOURCE = R"(
__kernel void gelu(__global const float* input, 
                   __global float* output, 
                   int size)
{
  const int idx = get_global_id(0);
  if (idx < size) 
  {
    const float x = input[idx];
    const float x_cubed = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * x_cubed);
    output[idx] = 0.5f * x * (1.0f + tanh(inner));
  }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
    std::vector<float> output(input.size());
    const size_t buffer_size = input.size() * sizeof(float);
    
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    if (!platforms.empty())
    {
        cl::Platform platform_obj = platforms[platform];
        std::vector<cl::Device> devices;
        platform_obj.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        
        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0]);
        
        cl::Program program(context, KERNEL_SOURCE);
        program.build(devices);
        
        cl::Kernel kernel(program, "gelu");
        
        cl::Buffer input_buffer(context, 
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                               buffer_size, 
                               const_cast<float*>(input.data()));
        
        cl::Buffer output_buffer(context, 
                                CL_MEM_WRITE_ONLY, 
                                buffer_size);
        
        kernel.setArg(0, input_buffer);
        kernel.setArg(1, output_buffer);
        kernel.setArg(2, static_cast<int>(input.size()));
        
        queue.enqueueNDRangeKernel(kernel, 
                                   cl::NullRange, 
                                   cl::NDRange(input.size()), 
                                   cl::NullRange);
        
        queue.enqueueReadBuffer(output_buffer, 
                                CL_TRUE, 
                                0, 
                                buffer_size, 
                                output.data());
    }
    
    return output;
}