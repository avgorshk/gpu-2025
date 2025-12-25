#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <string>
#include <stdexcept>

namespace {

const char* kGeluKernelSource = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          int n) {
    int idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    float x = input[idx];
    float xCubed = x * x * x;

    // sqrt(2/pi) constant
    const float kSqrt2OverPi = 0.7978845608f;
    const float kCoeff = 0.044715f;

    float inner = kSqrt2OverPi * (x + kCoeff * xCubed);
    
    // Compute tanh using exp: tanh(u) = (exp(2u) - 1) / (exp(2u) + 1)
    float exp2u = exp(2.0f * inner);
    float tanhVal = (exp2u - 1.0f) / (exp2u + 1.0f);
    
    output[idx] = 0.5f * x * (1.0f + tanhVal);
}
)";

struct OpenCLContext {
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    bool initialized = false;
};

OpenCLContext globalContext;

void initializeContext(int platformId) {
    if (globalContext.initialized) {
        return;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    if (platformId < 0 || platformId >= static_cast<int>(platforms.size())) {
        platformId = 0;
    }

    cl::Platform platform = platforms[platformId];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    
    if (devices.empty()) {
        throw std::runtime_error("No GPU devices found");
    }

    globalContext.device = devices[0];
    globalContext.context = cl::Context({globalContext.device});
    globalContext.queue = cl::CommandQueue(globalContext.context, globalContext.device);

    cl::Program program(globalContext.context, kGeluKernelSource);
    
    try {
        program.build({globalContext.device});
    } catch (...) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(globalContext.device);
        throw std::runtime_error("OpenCL kernel build failed: " + log);
    }

    globalContext.kernel = cl::Kernel(program, "gelu_kernel");
    globalContext.initialized = true;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    initializeContext(platform);

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    cl::Buffer inputBuffer(globalContext.context, 
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           bytes, 
                           const_cast<float*>(input.data()));
    
    cl::Buffer outputBuffer(globalContext.context, 
                            CL_MEM_WRITE_ONLY, 
                            bytes);

    globalContext.kernel.setArg(0, inputBuffer);
    globalContext.kernel.setArg(1, outputBuffer);
    globalContext.kernel.setArg(2, static_cast<int>(n));

    globalContext.queue.enqueueNDRangeKernel(globalContext.kernel, 
                                              cl::NullRange, 
                                              cl::NDRange(n), 
                                              cl::NullRange);
    globalContext.queue.finish();

    std::vector<float> output(n);
    globalContext.queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, bytes, output.data());

    return output;
}
