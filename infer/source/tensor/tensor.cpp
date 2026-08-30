#include "tensor/tensor.h"

namespace tensor
{
Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
            std::shared_ptr<base::DeviceAllocator> allocator, void* ptr): data_type_(data_type){
    size_= dim0;
    dims_.push_back(dim0);
    if(need_alloc && allocator){
        allocate_now(allocator);
    } else {
        if(ptr != nullptr){
            CHECK(need_alloc == false)
                << "When ptr is not nullptr, need_alloc should be false, but here is true";
            init_buffer(allocator, data_type, need_alloc, ptr);
        }
        else{
            LOG(INFO) << "Nullptr was passed in, please set need_alloc as true to avoid is_empty.";
        }
    }
};

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
                std::shared_ptr<base::DeviceAllocator> allocator, void* ptr): data_type_(data_type){
    size_= dim0 * dim1;
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    if(need_alloc && allocator){
        allocate_now(allocator);
    } else {
        init_buffer(allocator, data_type, need_alloc, ptr);
    }
};

bool Tensor::allocate_now(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc){
    if(!allocator){
        LOG(ERROR) << "Input is nullptr";
        return false;
    }
    size_t byte_size = this->byte_size();
    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
    return true;
};

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, 
    base::DataType data_type,bool need_alloc, void* ptr){
    if(!alloc && !need_alloc){
        std::shared_ptr<base::Buffer> buf = 
        std::make_shared<base::Buffer>(DataTypeSize(data_type)*size_, nullptr, ptr, true);
        this->buffer_ = buf;
    }else{
        allocate_now(alloc, true);
    }
};


// 接管 buffer 是否成功 ? 这需要 buffer 尺寸大于 tensor 的尺寸; 这不需要重新分配内存或者赋值数据
bool Tensor::assign(std::shared_ptr<base::Buffer> buffer){
    if (!buffer) {
        LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
        return false;
    }
    if (buffer_) {
        if (buffer_->get_device_type() != buffer->get_device_type()) {
        LOG(ERROR) << "The device type of the new buffer is different from the original one.";
        }
    }
    size_t this_size = this->byte_size();
    if(this_size > buffer->byte_size()){
        LOG(ERROR) << "The size of buffer is too small for the tensor!";
        return false;
    }
    buffer_ = buffer;
    return true;
};

void Tensor::to_cpu(){
    const base::DeviceType device_type = this->device_type();
    if(device_type == base::DeviceType::kDeviceUnknown){
        LOG(ERROR) << "kDataTypeUnknown\n";
    }else if(device_type == base::DeviceType::kDeviceCUDA){
        size_t byte_size = this->byte_size();
        auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
        auto buffer_cpu = std::make_shared<base::Buffer>(byte_size, alloc_cpu);
        alloc_cpu->memcpy(buffer_->ptr(), buffer_cpu->ptr(), byte_size, base::MemcpyKind::kMemcpyCUDA2CPU);
        this->buffer_ = buffer_cpu;
    }else{
        LOG(INFO) << "Tensor is already on cpu.\n";
    }
};
void Tensor::to_cuda(cudaStream_t stream){
    CHECK_NE(buffer_, nullptr);
    const base::DeviceType device_type = this->device_type();
    if(device_type == base::DeviceType::kDeviceUnknown){
        LOG(ERROR) << "kDataTypeUnknown\n";
    }else if(device_type == base::DeviceType::kDeviceCPU){
        size_t byte_size = this->byte_size();
        auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
        auto buffer = std::make_shared<base::Buffer>(byte_size, alloc);
        alloc->memcpy(buffer_->ptr(), buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA);
        this->buffer_ = buffer;
    }else{
        LOG(INFO) << "Tensor is already on cuda.\n";
    }
};

Tensor Tensor::clone() const{
    Tensor new_tensor = *this;
    auto allocator = buffer_->get_allocator();
    new_tensor.buffer_ = std::make_shared<base::Buffer>(this->byte_size(), allocator);
    new_tensor.buffer_->copy_from(buffer_.get());// get 是 std::shared_ptr 的成员函数，用来取得它内部保存的普通指针
    return new_tensor;
};

bool Tensor::is_empty() const{
    return size_==0 || buffer_==nullptr || buffer_->ptr()==nullptr; 
};

base::DeviceType Tensor::device_type() const{
    return buffer_->get_device_type();
};

size_t Tensor::byte_size() const { return this->size_ * DataTypeSize(data_type_); }
size_t Tensor::size() const{ return this->size_; };

std::shared_ptr<base::Buffer> Tensor::get_buffer() const{ return buffer_; };

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

} // namespace tensor
