#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <vector>
#include <stdexcept>

const char* GELU_KERNEL = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    float x = input[idx];
    const float coef = 0.044715f;
    const float scale = 0.7978845608f;
    float x3 = x * x * x;
    float inner = scale * (x + coef * x3);
    float tanh_val = tanh(inner);
    output[idx] = 0.5f * x * (1.0f + tanh_val);
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return std::vector<float>();
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    if (platform < 0 || platform >= (int)platforms.size()) {
        platform = 0;
    }

    std::vector<cl::Device> devices;
    platforms[platform].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No GPU devices found");
    }

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0]);

    cl::Program program(context, GELU_KERNEL);
    try {
        program.build(devices);
    }
    catch (...) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        throw std::runtime_error("Build failed: " + build_log);
    }

    cl::Kernel kernel(program, "gelu_kernel");

    size_t n = input.size();
    std::vector<float> output(n);
    cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        n * sizeof(float), const_cast<float*>(input.data()));
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float));

    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    kernel.setArg(2, (int)n);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);

    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, n * sizeof(float), output.data());

    return output;

}
