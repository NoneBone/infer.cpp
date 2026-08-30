#include "base/buffer.h"
#include <glog/logging.h>

namespace base {
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
                void *ptr, bool use_external)
    : byte_size_(byte_size),
        allocator_(allocator),
        ptr_(ptr),
        use_external_(use_external)
{
    if (!ptr_ && allocator_)
    {
        device_type_ = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size);
    }
};
Buffer::~Buffer() {
    if(!use_external_){
        allocator_->release(ptr_);
        ptr_ = nullptr;
    }
};

void Buffer::copy_from(const Buffer& src) const{// 引用写法，传入的是实际对象
    CHECK(allocator_ != nullptr);
    CHECK(ptr_ != nullptr);
    const DeviceType& src_device = src.device_type_;// & 表示引用, 采用 src_device 作为 右值的别名
    const DeviceType& dst_device = this->device_type_;
    if(src_device == DeviceType::kDeviceCPU && dst_device == DeviceType::kDeviceCUDA){
        // h2d    
        return allocator_->memcpy(src.ptr(), this->ptr_, byte_size_, MemcpyKind::kMemcpyCPU2CUDA);
    }else if(src_device == DeviceType::kDeviceCPU && dst_device == DeviceType::kDeviceCPU){
        // h2h
        return allocator_->memcpy(src.ptr(), this->ptr_, byte_size_, MemcpyKind::kMemcpyCPU2CPU);
    }else if(src_device == DeviceType::kDeviceCUDA && dst_device == DeviceType::kDeviceCPU){
        // d2h
        return allocator_->memcpy(src.ptr(), this->ptr_, byte_size_, MemcpyKind::kMemcpyCUDA2CPU);
    }else if(src_device == DeviceType::kDeviceCUDA && dst_device == DeviceType::kDeviceCUDA){
        // d2d
        return allocator_->memcpy(src.ptr(), this->ptr_, byte_size_, MemcpyKind::kMemcpyCUDACUDA);
    }else{
        LOG(INFO) << "unknown memcpy kind" << std::endl;
    }
};
void Buffer::copy_from(const Buffer* src) const{
    CHECK(src != nullptr);
    copy_from(*src);
};


size_t Buffer::byte_size() const {
  return byte_size_;
}

DeviceType Buffer::get_device_type() const{
    return device_type_;
};

void Buffer::set_device_type(DeviceType device_type){
    device_type_ = device_type;
};

std::shared_ptr<DeviceAllocator> Buffer::get_allocator() const {
  return allocator_;
}

void* Buffer::ptr(){
    return ptr_;
};

const void* Buffer::ptr() const {
    return ptr_;
};

bool Buffer::is_external() const {
    return use_external_;
};

} // namespace base
