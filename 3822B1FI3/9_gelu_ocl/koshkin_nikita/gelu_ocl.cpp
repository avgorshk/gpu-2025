#include "gelu_ocl.h"

#define BLOCK_SIZE 256

const char* GeluKernel = R"(
    __kernel void GeluKernel(__global const float* in, __global float* out, const int size) {
        int idx = get_global_id(0);
        if (idx < size) {
            float x = in[idx];
            out[idx] = 0.5f * x * (1.0f + tanh(0.797884f * (x + 0.044715f * (x * x * x))));
        }
    }
    )";


std::vector<float> GeluOCL(const std::vector<float>& input, int platform_index) {

    const size_t size = input.size(), memory = size * sizeof(float);
    std::vector<float> result(size);

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem in, out;
    cl_int st;
    
    cl_uint nplat = 0;
    clGetPlatformIDs(0, nullptr, &nplat);
    if (nplat == 0) {
		throw std::runtime_error("No OpenCL platforms");
	}
    if (platform_index < 0 || (cl_uint)platform_index >= nplat){
        throw std::runtime_error("platform_index out of range");
	}
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    platform = plats[(size_t)platform_index];
	
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &st);
    if (st != CL_SUCCESS) {
		throw std::runtime_error("clCreateContext failed");
	}
	
    queue = clCreateCommandQueue(context, device, 0, &st);
    if (st != CL_SUCCESS) {
		throw std::runtime_error("clCreateCommandQueue failed");
	}

    program = clCreateProgramWithSource(context, 1, &GeluKernel, NULL, &st);
    if (st != CL_SUCCESS) {
		throw std::runtime_error("clCreateProgramWithSource failed");
	}
    const char* args = "-cl-fast-relaxed-math";
    st = clBuildProgram(program, 1, &device, args, NULL, NULL);
    if (st != CL_SUCCESS) {
        size_t loglen = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &loglen);
        std::string log(loglen, '\0');
        if (loglen) {
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, loglen, log.data(), nullptr);
		}
        throw std::runtime_error(std::string("clBuildProgram failed: ") + log);
    }
    kernel = clCreateKernel(program, "GeluKernel", &st);
    if (st != CL_SUCCESS) {
		throw std::runtime_error("clCreateKernel failed");
	}

    in  = clCreateBuffer(context, CL_MEM_READ_ONLY,  memory, NULL, &st);
    out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, memory, NULL, &st);
    if (st != CL_SUCCESS) {
		throw std::runtime_error("clCreateBuffer failed");
	}

    clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, memory, input.data(), 0, NULL, NULL);

    cl_uint n = (cl_uint)size;
    clSetKernelArg(kernel, 0, sizeof(cl_mem),  &in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem),  &out);
    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n);

    const size_t block = BLOCK_SIZE;
    const size_t grid  = ((size + block - 1) / block) * block;

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &grid, &block, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, out, CL_TRUE, 0, memory, result.data(), 0, NULL, NULL);

    clReleaseMemObject(in);
    clReleaseMemObject(out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return result;
}