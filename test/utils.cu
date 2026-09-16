#include "glog/logging.h"
#include "utils.cuh"

__global__ void test_function_cu(float* cu_arr,int32_t size, float value){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= size){
        return;
    }
    cu_arr[tid] = value;
}

void set_value_cu(float* dst, int size, float val){
    int threadNum = 256;
    int blockNum = CEIL(size, threadNum);
    test_function_cu<<<blockNum, threadNum>>>(dst, size, val);
    cudaDeviceSynchronize();
    cudaError_t state = cudaGetLastError();
    CHECK_EQ(state, cudaSuccess);
}

__global__ void set_value_fp16_kernel(__half* dst, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(val);
    }
}

void set_value_cu_fp16(__half* dst, int size, float val) {
    int block = 256;
    int grid = (size + block - 1) / block;

    set_value_fp16_kernel<<<grid, block>>>(dst, size, val);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}