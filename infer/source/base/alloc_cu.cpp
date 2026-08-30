#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base{

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA){};

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    if(!byte_size){
        return nullptr;
    }
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess);
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    CHECK(state == cudaSuccess);
    return ptr;
};

void CUDADeviceAllocator::release(void* ptr) const{
    if (!ptr) {
        return;
    }
    cudaError_t state = cudaSuccess;
    if(ptr) state = cudaFree(ptr);
    CHECK(state == cudaSuccess);
};

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance =nullptr;
}// namespace base