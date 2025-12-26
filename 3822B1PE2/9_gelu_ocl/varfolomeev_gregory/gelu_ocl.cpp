#include "gelu_ocl.h"
#include <CL/cl.h>
#include <mutex>

static cl_context g_context = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_kernel g_kernel = nullptr;
static cl_program g_program = nullptr;
static bool g_initialized = false;
static std::mutex g_init_mutex;
static int g_cached_platform = -1;

const char* GELU_KERNEL_SOURCE = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float t = 0.7978845608028654f * (x + 0.044715f * x3);
    float exp_neg2t = exp(-2.0f * t);
    float tanh_t = (1.0f - exp_neg2t) / (1.0f + exp_neg2t);
    output[idx] = 0.5f * x * (1.0f + tanh_t);
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }
    
    int n = input.size();
    std::vector<float> output(n);
    
    // Initialize OpenCL resources once (boilerplate code optimization)
    std::lock_guard<std::mutex> lock(g_init_mutex);
    
    if (!g_initialized || g_cached_platform != platform) {
        // Cleanup previous resources if platform changed
        if (g_initialized) {
            clReleaseKernel(g_kernel);
            clReleaseProgram(g_program);
            clReleaseCommandQueue(g_queue);
            clReleaseContext(g_context);
        }
        
        // Get platforms
        cl_uint num_platforms;
        clGetPlatformIDs(0, NULL, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), NULL);
        
        // Get GPU device from specified platform (device 0)
        cl_device_id device;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        
        // Create context and command queue
        g_context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        g_queue = clCreateCommandQueue(g_context, device, 0, NULL);
        
        // Create program and build
        const char* sources[] = { GELU_KERNEL_SOURCE };
        g_program = clCreateProgramWithSource(g_context, 1, sources, NULL, NULL);
        clBuildProgram(g_program, 1, &device, NULL, NULL, NULL);
        
        // Create kernel
        g_kernel = clCreateKernel(g_program, "gelu_kernel", NULL);
        
        g_initialized = true;
        g_cached_platform = platform;
    }
    
    size_t bytes = n * sizeof(float);
    
    // Create buffers
    cl_mem input_buffer = clCreateBuffer(g_context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem output_buffer = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    // Async write for overlap
    cl_event write_event;
    clEnqueueWriteBuffer(g_queue, input_buffer, CL_FALSE, 0, bytes, input.data(), 0, NULL, &write_event);
    
    // Set kernel arguments
    clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(g_kernel, 2, sizeof(int), &n);
    
    // Execute kernel with async operation
    size_t global_size = ((n + 255) / 256) * 256;
    size_t local_size = 256;
    cl_event kernel_event;
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, NULL, &global_size, &local_size, 1, &write_event, &kernel_event);
    
    // Async read for overlap
    cl_event read_event;
    clEnqueueReadBuffer(g_queue, output_buffer, CL_FALSE, 0, bytes, output.data(), 1, &kernel_event, &read_event);
    clWaitForEvents(1, &read_event);
    
    // Cleanup events and buffers
    clReleaseEvent(read_event);
    clReleaseEvent(kernel_event);
    clReleaseEvent(write_event);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    
    return output;
}


