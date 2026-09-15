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
    if(enabled_ && byte_size > 1ULL * 1024 * 1024){
        // v1 big Small buffer
        auto& _buffers =big_buffers_map_[id];
        int sel_id = -1;
        int i=0;
        for(; i<_buffers.size();i++){
            auto &b = _buffers[i];
            bool suitable = !b.busy && b.byte_size >= byte_size;
            bool close_enough = byte_size < 1024 * 1024? true : (b.byte_size - byte_size) < 1024 * 1024;
            if((suitable && close_enough) && 
                (sel_id < 0 || _buffers[sel_id].byte_size > b.byte_size)){
                sel_id = i;
            }
        }


        if(sel_id != -1){
            _buffers[sel_id].busy = true;
            if (byte_size <= 1ULL * 1024 * 1024){
                if (no_busy_cnt_[id] >= _buffers[sel_id].byte_size) {
                    no_busy_cnt_[id] -= _buffers[sel_id].byte_size;
                } else {
                    no_busy_cnt_[id] = 0;
                }
            }
            return _buffers[sel_id].data;
        }
    }
    if(enabled_ && byte_size <= 1024*1024){
        auto& small_buffers = small_buffers_map_[id];
        size_t i = 0;
        for (; i < small_buffers.size(); ++i) {
            if(small_buffers[i].byte_size >= byte_size && !small_buffers[i].busy){
                small_buffers[i].busy = true;
                if(no_busy_cnt_[id] >= small_buffers[i].byte_size){
                    no_busy_cnt_[id] -= small_buffers[i].byte_size;
                }else{
                    no_busy_cnt_[id] = 0;
                }
                return small_buffers[i].data;
            }
        }
    }
    
    // v0 naive malloc directly
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    CHECK(state == cudaSuccess);
    if(enabled_){
        auto& _buffers = (byte_size > 1024 * 1024)? big_buffers_map_[id] : small_buffers_map_[id];
        _buffers.push_back({ptr, byte_size, true});
    }
    return ptr;
};

void CUDADeviceAllocator::release(void* ptr) const{
    if (!ptr) {
        return;
    }
    if(enabled_ && small_buffers_map_.empty()){
        return;
    }
    if(enabled_){
        for(auto& it : small_buffers_map_){
            auto dev_id = it.first;
            if(no_busy_cnt_[dev_id] > 1024 * 1024 * 1024){// 存在大量空闲，忙碌者保留，空闲者清楚
                auto& cuda_buffers = it.second;
                std::vector<CudaMemoryBuffer> kept;
                for(int i=0; i< cuda_buffers.size();i++){
                    if(cuda_buffers[i].busy){
                        kept.push_back(cuda_buffers[i]);
                    }else{
                        cudaError_t sta = cudaSetDevice(dev_id);
                        sta = cudaFree(cuda_buffers[i].data);
                        CHECK(sta == cudaSuccess);
                    }
                }
                cuda_buffers.clear();
                it.second = kept;
                no_busy_cnt_[dev_id] = 0; 
            }
        }
        for(auto& it : small_buffers_map_){
            auto& cuda_buffers = it.second;
            for(int i=0; i< cuda_buffers.size();i++){
                if(cuda_buffers[i].data == ptr){
                    no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;// 统计小 buffer
                    cuda_buffers[i].busy = false;
                    return;
                }
            }
            auto& big_buffers = big_buffers_map_[it.first];
            for(int i=0; i< big_buffers.size();i++){
                if(big_buffers[i].data == ptr){
                    big_buffers[i].busy = false;
                    return;
                }
            }
        }
    }
    cudaError_t state = cudaSuccess;
    if(ptr) state = cudaFree(ptr);
    CHECK(state == cudaSuccess);
};

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance =nullptr;
}// namespace base