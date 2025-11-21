#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <string>

static const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input,
                         __global float* output,
                         const int num_elements)
{
    int idx = get_global_id(0);
    if (idx < num_elements) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanh(inner));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return input;
    }

    static struct OpenCLState {
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
        bool initialized = false;

        ~OpenCLState() {
            if (kernel) clReleaseKernel(kernel);
            if (program) clReleaseProgram(program);
            if (queue) clReleaseCommandQueue(queue);
            if (context) clReleaseContext(context);
        }
    } state;

    if (!state.initialized) {
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_platform_id selected_platform = platforms[platform];

        cl_uint num_devices;
        clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        cl_device_id device = devices[0];

        state.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        state.queue = clCreateCommandQueue(state.context, device, 0, nullptr);
        state.program = clCreateProgramWithSource(state.context, 1, &gelu_kernel_source, nullptr, nullptr);

        clBuildProgram(state.program, 1, &device, nullptr, nullptr, nullptr);
        state.kernel = clCreateKernel(state.program, "gelu_kernel", nullptr);

        state.initialized = true;
    }

    const size_t num_elements = input.size();
    const size_t data_bytes = num_elements * sizeof(float);

    cl_mem input_buffer = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                        data_bytes, (void*)input.data(), nullptr);
    cl_mem output_buffer = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, data_bytes, nullptr, nullptr);

    int elements_int = static_cast<int>(num_elements);
    clSetKernelArg(state.kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(state.kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(state.kernel, 2, sizeof(int), &elements_int);

    size_t global_work_size = (num_elements + 255) / 256 * 256;
    clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);

    std::vector<float> output(num_elements);
    clEnqueueReadBuffer(state.queue, output_buffer, CL_TRUE, 0, data_bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);

    return output;
}
