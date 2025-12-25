#include "gelu_ocl.h"

#include <CL/cl.h>
#include <vector>

static const char kGeluKernelSource[] = R"CLC(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float x2 = x * x;
        float x3 = x2 * x;
        float u = 0.7978845608f * (x + 0.044715f * x3);
        float t = tanh(u);
        output[i] = 0.5f * x * (1.0f + t);
    }
}
)CLC";

struct OpenCLState {
    int platform_index = -1;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    bool initialized = false;

    void Release() {
        if (kernel) {
            clReleaseKernel(kernel);
            kernel = nullptr;
        }
        if (program) {
            clReleaseProgram(program);
            program = nullptr;
        }
        if (queue) {
            clReleaseCommandQueue(queue);
            queue = nullptr;
        }
        if (context) {
            clReleaseContext(context);
            context = nullptr;
        }
        platform = nullptr;
        device = nullptr;
        initialized = false;
    }

    ~OpenCLState() {
        Release();
    }
};

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    std::vector<float> output(input.size());
    if (input.empty()) {
        return output;
    }

    static OpenCLState state;
    if (!state.initialized || state.platform_index != platform) {
        state.Release();

        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        if (num_platforms == 0 || platform < 0 || platform >= static_cast<int>(num_platforms)) {
            return output;
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        state.platform = platforms[platform];
        state.platform_index = platform;

        cl_uint num_devices = 0;
        clGetDeviceIDs(state.platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (num_devices == 0) {
            return output;
        }

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(state.platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        state.device = devices[0];

        cl_int err = CL_SUCCESS;
        state.context = clCreateContext(nullptr, 1, &state.device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            return output;
        }

        state.queue = clCreateCommandQueue(state.context, state.device, 0, &err);
        if (err != CL_SUCCESS) {
            state.Release();
            return output;
        }

        const char* sources[] = {kGeluKernelSource};
        state.program = clCreateProgramWithSource(state.context, 1, sources, nullptr, &err);
        if (err != CL_SUCCESS) {
            state.Release();
            return output;
        }

        err = clBuildProgram(state.program, 1, &state.device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            state.Release();
            return output;
        }

        state.kernel = clCreateKernel(state.program, "gelu_kernel", &err);
        if (err != CL_SUCCESS) {
            state.Release();
            return output;
        }

        state.initialized = true;
    }

    size_t n = input.size();
    size_t bytes = n * sizeof(float);
    cl_int err = CL_SUCCESS;
    cl_mem input_buf = clCreateBuffer(state.context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        return output;
    }
    cl_mem output_buf = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buf);
        return output;
    }

    err = clEnqueueWriteBuffer(state.queue, input_buf, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buf);
        clReleaseMemObject(output_buf);
        return output;
    }

    int n_int = static_cast<int>(n);
    err = clSetKernelArg(state.kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(state.kernel, 1, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(state.kernel, 2, sizeof(int), &n_int);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buf);
        clReleaseMemObject(output_buf);
        return output;
    }

    size_t local = 256;
    size_t global = (n + local - 1) / local * local;
    err = clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(input_buf);
        clReleaseMemObject(output_buf);
        return output;
    }

    clEnqueueReadBuffer(state.queue, output_buf, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);

    return output;
}
