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