#include "gelu_ocl.h"
#include <CL/cl.hpp>
#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform_id) {
    if (input.empty()) {
        return {};
    }

    size_t n = input.size();

    std::vector<cl::Platform> platforms;
    if (cl::Platform::get(&platforms) != CL_SUCCESS) {
        return std::vector<float>(n, 0.0f);
    }

    if (platform_id < 0 || static_cast<size_t>(platform_id) >= platforms.size()) {
        return std::vector<float>(n, 0.0f);
    }

    cl::Platform platform = platforms[platform_id];
    std::vector<cl::Device> devices;
    if (platform.getDevices(CL_DEVICE_TYPE_GPU, &devices) != CL_SUCCESS || devices.empty()) {
        return std::vector<float>(n, 0.0f);
    }

    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    const char* kernel_source = R"(
        __kernel void gelu_ocl(__global const float* input,
                               __global float* output,
                               const uint n) {
            const uint idx = get_global_id(0);
            if (idx >= n) return;

            const float x = input[idx];
            const float x3 = x * x * x;
            const float SQRT_2_OVER_PI = 0.7978845608f;
            const float GELU_COEFF = 0.044715f;
            const float MUL = 2.0f * SQRT_2_OVER_PI;

            const float z = MUL * (x + GELU_COEFF * x3);
            output[idx] = x / (1.0f + native_exp(-z));
        }
    )";

    cl::Program program(context, kernel_source);
    if (program.build({ device }) != CL_SUCCESS) {
        return std::vector<float>(n, 0.0f);
    }

    cl::Buffer input_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float), const_cast<float*>(input.data()));
    cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, n * sizeof(float));

    cl::Kernel kernel(program, "gelu_ocl");
    kernel.setArg(0, input_buf);
    kernel.setArg(1, output_buf);
    kernel.setArg(2, static_cast<cl_uint>(n));

    cl::NDRange global_size((n + 255) / 256 * 256);
    cl::NDRange local_size(256);

    if (queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size) != CL_SUCCESS) {
        return std::vector<float>(n, 0.0f);
    }

    queue.finish();

    std::vector<float> output(n);
    if (queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, n * sizeof(float), output.data()) != CL_SUCCESS) {
        return std::vector<float>(n, 0.0f);
    }

    return output;
}