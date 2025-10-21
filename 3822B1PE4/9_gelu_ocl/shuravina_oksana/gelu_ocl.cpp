#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

// Константа для формулы GELU
#define M_SQRT2_OVER_PI 0.7978845608028654f
#define GELU_COEF 0.044715f

// Код ядра OpenCL
const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, 
                         __global float* output, 
                         int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        // GELU формула: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // Оптимизированная версия через exp для лучшей производительности
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        float tanh_val = 1.0f - 2.0f / (exp(2.0f * inner) + 1.0f);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

// Альтернативная оптимизированная версия с одной экспонентой
__kernel void gelu_kernel_optimized(__global const float* input, 
                                   __global float* output, 
                                   int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        // Более быстрая аппроксимация: x * σ(1.702x)
        // где σ - сигмоида: 1 / (1 + exp(-x))
        output[idx] = x * (1.0f / (1.0f + exp(-1.702f * x)));
    }
}

// Еще одна точная версия с использованием erf
__kernel void gelu_kernel_erf(__global const float* input, 
                             __global float* output, 
                             int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        // Точная формула: 0.5 * x * (1 + erf(x / sqrt(2)))
        // Но в OpenCL нет встроенной erf, поэтому используем аппроксимацию
        float z = x * 0.7071067811865475f; // x / sqrt(2)
        // Аппроксимация erf через tanh
        float erf_approx = tanh(0.7978845608028654f * z * (1.0f + 0.044715f * z * z));
        output[idx] = 0.5f * x * (1.0f + erf_approx);
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return input;
    }

    cl_int err;
    cl_uint num_platforms;
    cl_platform_id* platforms = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem input_buffer = nullptr;
    cl_mem output_buffer = nullptr;

    try {
        // Получение количества платформ
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            throw std::runtime_error("No OpenCL platforms found");
        }

        // Получение списка платформ
        platforms = new cl_platform_id[num_platforms];
        err = clGetPlatformIDs(num_platforms, platforms, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get platform IDs");
        }

        // Проверка индекса платформы
        if (platform < 0 || platform >= static_cast<int>(num_platforms)) {
            throw std::runtime_error("Invalid platform index");
        }

        cl_platform_id selected_platform = platforms[platform];

        // Получение GPU устройства
        err = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
        if (err != CL_SUCCESS) {
            // Попробуем получить любое устройство, если GPU не найдено
            err = clGetDeviceIDs(selected_platform, CL_DEVICE_TYPE_ALL, 1, &device_id, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to get OpenCL device");
            }
        }

        // Создание контекста
        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL context");
        }

        // Создание командной очереди
        command_queue = clCreateCommandQueue(context, device_id, 0, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create command queue");
        }

        // Создание программы
        program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create program");
        }

        // Компиляция программы
        err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Получение логов компиляции в случае ошибки
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::string error_msg = "Failed to build program: " + std::string(log.data());
            throw std::runtime_error(error_msg);
        }

        // Создание ядра (используем оптимизированную версию)
        kernel = clCreateKernel(program, "gelu_kernel_optimized", &err);
        if (err != CL_SUCCESS) {
            // Попробуем другую версию ядра
            kernel = clCreateKernel(program, "gelu_kernel", &err);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to create kernel");
            }
        }

        // Создание буферов
        size_t data_size = input.size() * sizeof(float);
        input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                     data_size, (void*)input.data(), &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create input buffer");
        }

        output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create output buffer");
        }

        // Установка аргументов ядра
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &input.size());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to set kernel arguments");
        }

        // Выполнение ядра
        size_t global_size = input.size();
        size_t local_size;
        
        // Получение рекомендуемого размера рабочей группы
        err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, 
                                      sizeof(local_size), &local_size, nullptr);
        if (err != CL_SUCCESS || local_size == 0) {
            local_size = 256; // Значение по умолчанию
        }
        
        // Ограничение размера рабочей группы
        if (local_size > 256) {
            local_size = 256;
        }
        
        // Выравнивание глобального размера
        global_size = ((input.size() + local_size - 1) / local_size) * local_size;

        err = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, 
                                    &global_size, &local_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to execute kernel");
        }

        // Ожидание завершения вычислений
        err = clFinish(command_queue);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to finish command queue");
        }

        // Чтение результата
        std::vector<float> output(input.size());
        err = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, 
                                 data_size, output.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read output buffer");
        }

        // Освобождение ресурсов
        if (output_buffer) clReleaseMemObject(output_buffer);
        if (input_buffer) clReleaseMemObject(input_buffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (command_queue) clReleaseCommandQueue(command_queue);
        if (context) clReleaseContext(context);
        if (platforms) delete[] platforms;

        return output;

    } catch (const std::exception& e) {
        // Освобождение ресурсов в случае ошибки
        if (output_buffer) clReleaseMemObject(output_buffer);
        if (input_buffer) clReleaseMemObject(input_buffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (command_queue) clReleaseCommandQueue(command_queue);
        if (context) clReleaseContext(context);
        if (platforms) delete[] platforms;
        
        throw;
    }
}