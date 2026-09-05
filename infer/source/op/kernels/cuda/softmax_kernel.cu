#include "softmax_kernel.cuh"
#include "../kernels_interface.h"

namespace kernel {

#include <cfloat>
#include <cuda_runtime.h>
#include <stdint.h>

inline __device__ __host__ unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val,
            __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void set_max_and_sum_kernel(
    float *max_values,
    float *sum_values,
    int rows
){
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < rows) {
        max_values[idx] = -FLT_MAX;
        sum_values[idx] = 0.0;
    }
}

__global__ void max_kernel(
    const float *x,
    int rows,
    int cols,
    int coarseFactor,
    float *max_values
){
    extern __shared__ float smem[];
    auto tid = threadIdx.x;
    auto row = blockIdx.y;

    float max_value = -FLT_MAX;
    for (int i = 0; i < coarseFactor; i++) {
        auto col =blockIdx.x * (blockDim.x * 
        coarseFactor) + tid + i * blockDim.x;
        if (col < cols) {
            max_value =
                fmaxf(max_value, x[(row * cols) + col]);
        }
    }
    smem[tid] = max_value;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(&max_values[row], smem[0]);
    }
}

__global__ void sum_kernel(
    const float *x,
    float* y,
    int rows,
    int cols,
    int coarseFactor,
    const float *max_values,
    float *sum_values
){
    extern __shared__ float smem[];
    auto tid = threadIdx.x;
    auto row = blockIdx.y;

    float sum = 0.0f;
    for (int i = 0; i < coarseFactor; i++) {
        auto col =
            blockIdx.x * (blockDim.x * coarseFactor) + tid + i * blockDim.x;
        if (col < cols) {
            float val =
                expf(x[(row * cols) + col] - max_values[row]);
            y[(row * cols) + col] = val;
            sum += val;
        }
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&sum_values[row], smem[0]);
    }
}

__global__ void softmax_kernel(
    float *y,
    int rows,
    int cols,
    const float *sum_values
){
    auto tid = threadIdx.x;
    auto row = blockIdx.y;
    auto col = tid + blockIdx.x * blockDim.x;

    if (col < cols) {
        y[row * cols + col] /= sum_values[row];
    }
}

void softmax_inplace_cu(const tensor::Tensor& input, void* stream) {// TODO: support stream
    CHECK_EQ(input.is_empty(), false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

    int32_t size = static_cast<int32_t>(input.size());
    const float* input_ptr = input.ptr<float>();

    auto rows = 1;
    auto cols = size;

    auto coarseFactor = 32;
    dim3 threads(256);
    dim3 blocks(cdiv(cols, threads.x * coarseFactor), rows);
    auto smem_size = threads.x * sizeof(float);

    float* max_values = nullptr;
    cudaMalloc(&max_values, rows * sizeof(float));
    float* sum_values = nullptr;
    cudaMalloc(&sum_values, rows * sizeof(float));
    set_max_and_sum_kernel<<<cdiv(rows, threads.x),
        threads>>>(max_values, sum_values, rows);
    max_kernel<<<blocks, threads, smem_size>>>(input_ptr, rows,
        cols, coarseFactor, max_values);
    sum_kernel<<<blocks, threads, smem_size>>>(input_ptr, const_cast<float*>(input_ptr),
        rows, cols, coarseFactor, max_values, sum_values);
    auto blocks_softmax = dim3(cdiv(cols, threads.x), rows);
    softmax_kernel<<<blocks_softmax, threads>>>(const_cast<float*>(input_ptr), rows,
        cols, sum_values);
}
} // kernel