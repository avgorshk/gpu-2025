#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include "gelu_ocl.h"

#include <mutex>

namespace {

const char kGeluProgram[] = R"CLC(
__kernel void gelu_eval(__global const float* input,
                        __global float* output,
                        const int count) {
    const int gid = get_global_id(0);
    if (gid >= count) {
        return;
    }

    const float x = input[gid];
    const float cubic = x * x * x;
    const float term = 0.7978845608f * (x + 0.044715f * cubic);
    const float exp_val = exp(-2.0f * term);
    const float sigmoid = 1.0f / (1.0f + exp_val);
    output[gid] = x * sigmoid;
}
)CLC";

struct OpenClPipeline {
    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    cl::Kernel kernel;
};

OpenClPipeline& get_pipeline(int platform_index) {
    static OpenClPipeline pipeline;
    static int current_platform = -1;
    static std::once_flag once;
    static std::mutex guard;

    auto initialise = [&](int platform_id) {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw cl::Error(CL_INVALID_PLATFORM, "no OpenCL platforms");
        }

        if (platform_id < 0 || platform_id >= static_cast<int>(platforms.size())) {
            platform_id = 0;
        }

        std::vector<cl::Device> devices;
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw cl::Error(CL_DEVICE_NOT_FOUND, "no GPU devices on selected platform");
        }

        pipeline.device = devices.front();
        pipeline.context = cl::Context(pipeline.device);
        pipeline.queue = cl::CommandQueue(pipeline.context, pipeline.device);

        cl::Program program(pipeline.context, kGeluProgram);
        program.build({pipeline.device});
        pipeline.kernel = cl::Kernel(program, "gelu_eval");
        current_platform = platform_id;
    };

    if (current_platform == -1) {
        std::lock_guard<std::mutex> lock(guard);
        if (current_platform == -1) {
            initialise(platform_index);
        }
    } else if (platform_index != current_platform) {
        std::lock_guard<std::mutex> lock(guard);
        if (platform_index != current_platform) {
            initialise(platform_index);
        }
    }

    return pipeline;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    try {
        OpenClPipeline& pipe = get_pipeline(platform);
        const std::size_t bytes = input.size() * sizeof(float);

        cl::Buffer in_buffer(pipe.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             bytes, const_cast<float*>(input.data()));
        cl::Buffer out_buffer(pipe.context, CL_MEM_WRITE_ONLY, bytes);

        pipe.kernel.setArg(0, in_buffer);
        pipe.kernel.setArg(1, out_buffer);
        pipe.kernel.setArg(2, static_cast<int>(input.size()));

        const cl::NDRange global_range(input.size());
        pipe.queue.enqueueNDRangeKernel(pipe.kernel, cl::NullRange, global_range, cl::NullRange);
        pipe.queue.finish();

        std::vector<float> result(input.size());
        pipe.queue.enqueueReadBuffer(out_buffer, CL_TRUE, 0, bytes, result.data());
        return result;
    } catch (const cl::Error&) {
        return {};
    }
}
