#include "gelu_ocl.h"

static const char* oclSource = R"CLC(
__kernel void gelu_exec(__global const float* src,
                        __global float* dst,
                        const int count)
{
    int gid = get_global_id(0);
    if (gid < count) {
        const float val = src[gid];
        const float coeff = 0.044715f;
        const float k = 0.7978845608028654f; // sqrt(2/pi)
        float cube = val * val * val;
        float t = k * (val + coeff * cube);
        float sigm = 1.0f / (1.0f + exp(-2.0f * t));
        dst[gid] = val * sigm;
    }
}
)CLC";

struct OclEnv {
  cl_platform_id plat;
  cl_device_id dev;
  cl_context ctx;
  cl_command_queue cmd;
  cl_program prog;
  cl_kernel kern;
};

OclEnv* InitOclEnv(int platformId) {
  OclEnv* cl_env = new OclEnv();

  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount);
  std::vector<cl_platform_id> platformList(platformCount);
  clGetPlatformIDs(platformCount, platformList.data(), nullptr);
  cl_env->plat = platformList[platformId];

  cl_uint deviceCount = 0;
  clGetDeviceIDs(cl_env->plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
  std::vector<cl_device_id> deviceList(deviceCount);
  clGetDeviceIDs(cl_env->plat, CL_DEVICE_TYPE_GPU, deviceCount,
                 deviceList.data(), nullptr);
  cl_env->dev = deviceList[0];

  cl_env->ctx =
      clCreateContext(nullptr, 1, &cl_env->dev, nullptr, nullptr, nullptr);
  cl_env->cmd = clCreateCommandQueue(cl_env->ctx, cl_env->dev, 0, nullptr);

  const char* src_ptr = oclSource;
  size_t src_len = std::strlen(src_ptr);
  cl_env->prog =
      clCreateProgramWithSource(cl_env->ctx, 1, &src_ptr, &src_len, nullptr);
  clBuildProgram(cl_env->prog, 1, &cl_env->dev, nullptr, nullptr, nullptr);
  cl_env->kern = clCreateKernel(cl_env->prog, "gelu_exec", nullptr);

  return cl_env;
}

void ReleaseOclEnv(OclEnv* env) {
  if (!env) return;
  clReleaseKernel(env->kern);
  clReleaseProgram(env->prog);
  clReleaseCommandQueue(env->cmd);
  clReleaseContext(env->ctx);
  delete env;
}

std::vector<float> RunGeluKernel(OclEnv* env,
                                 const std::vector<float>& srcData) {
  size_t total = srcData.size();
  std::vector<float> result(total);
  size_t dataSize = total * sizeof(float);

  cl_mem bufIn =
      clCreateBuffer(env->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     dataSize, (void*)srcData.data(), nullptr);
  cl_mem bufOut =
      clCreateBuffer(env->ctx, CL_MEM_WRITE_ONLY, dataSize, nullptr, nullptr);

  int count = static_cast<int>(total);
  clSetKernelArg(env->kern, 0, sizeof(cl_mem), &bufIn);
  clSetKernelArg(env->kern, 1, sizeof(cl_mem), &bufOut);
  clSetKernelArg(env->kern, 2, sizeof(int), &count);

  const size_t localSize = 256;
  size_t globalSize = ((total + localSize - 1) / localSize) * localSize;

  clEnqueueNDRangeKernel(env->cmd, env->kern, 1, nullptr, &globalSize,
                         &localSize, 0, nullptr, nullptr);
  clEnqueueReadBuffer(env->cmd, bufOut, CL_TRUE, 0, dataSize, result.data(), 0,
                      nullptr, nullptr);

  clReleaseMemObject(bufIn);
  clReleaseMemObject(bufOut);

  return result;
}

std::vector<float> GeluOCL(const std::vector<float>& data, int platformId = 0) {
  static OclEnv* persistentEnv = InitOclEnv(platformId);
  return RunGeluKernel(persistentEnv, data);
}
