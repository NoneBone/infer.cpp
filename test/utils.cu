#include "glog/logging.h"
#include "utils.cuh"

__global__ void test_function_cu(float* cu_arr,int32_t size, float value){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= size){
        return;
    }
    cu_arr[tid] = value;
}

void test_function(float* arr,int32_t size, float value){
    if(!arr){
        return;
    }
}

void set_value_cu(){
    return;
}