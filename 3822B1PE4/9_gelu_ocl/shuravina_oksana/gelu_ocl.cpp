#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <stdexcept>
#include <string>

static const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input,
                         __global float* output,
                         const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        // Быстрая аппроксимация GELU
        output[i] = 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return input;

    static struct OpenCLState {
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;
        bool initialized = false;
        
        ~OpenCLState() {
            if (initialized) {
                if (kernel) clReleaseKernel(kernel);
                if (program) clReleaseProgram(program);
                if (queue) clReleaseCommandQueue(queue);
                if (context) clReleaseContext(context);
            }
        }
    } state;

    if (!state.initialized) {
        cl_int err;
        
        // Получение платформы
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        if (platform < 0 || platform >= static_cast<int>(num_platforms)) {
            throw std::runtime_error("Invalid platform index");
        }
        
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        state.platform = platforms[platform];
        
        // Получение GPU устройства
        cl_uint num_devices;
        clGetDeviceIDs(state.platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (num_devices == 0) throw std::runtime_error("No GPU devices found");
        
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(state.platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        state.device = devices[0];
        
        // Создание контекста и очереди
        state.context = clCreateContext(nullptr, 1, &state.device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create context");
        
        state.queue = clCreateCommandQueue(state.context, state.device, 0, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");
        
        // Компиляция программы
        state.program = clCreateProgramWithSource(state.context, 1, &gelu_kernel_source, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");
        
        err = clBuildProgram(state.program, 1, &state.device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to build program");
        
        state.kernel = clCreateKernel(state.program, "gelu_kernel", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create kernel");
        
        state.initialized = true;
    }

    // Выполнение
    size_t n = input.size();
    std::vector<float> output(n);
    size_t bytes = n * sizeof(float);
    
    cl_mem input_buf = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                     bytes, (void*)input.data(), nullptr);
    cl_mem output_buf = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
    
    clSetKernelArg(state.kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(state.kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(state.kernel, 2, sizeof(int), &n);
    
    size_t global = (n + 255) / 256 * 256;
    clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(state.queue, output_buf, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    
    return output;
}