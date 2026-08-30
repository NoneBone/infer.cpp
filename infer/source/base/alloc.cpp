#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base{
// 统一的拷贝接口, 对于 cpu 和 cuda
void DeviceAllocator::memcpy(const void *src_ptr, void *dst_ptr, size_t byte_size,
                    MemcpyKind memcpy_kind, void *stream, bool need_sync) const {
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dst_ptr, nullptr);
    if(!byte_size){
        return;
    }
    // copy
    if(memcpy_kind == MemcpyKind::kMemcpyCPU2CPU){
        std::memcpy(dst_ptr, src_ptr, byte_size);
    }else if(memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA){
        cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    }else if(memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU){
        cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else if(memcpy_kind == MemcpyKind::kMemcpyCUDACUDA){
        cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else{
        LOG(INFO) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
    if(need_sync){
        cudaDeviceSynchronize();
    }
};
    
}// namespace base