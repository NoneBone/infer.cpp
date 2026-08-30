#ifndef INFER_INCLUDE_TENSOR_H
#define INFER_INCLUDE_TENSOR_H
#include "cuda_runtime_api.h"
#include "base/base.h"
#include "base/buffer.h"
#include "glog/logging.h"

namespace tensor{
class Tensor{
private:
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    std::vector<int32_t> dims_;
    size_t size_ = 0;
    std::shared_ptr<base::Buffer> buffer_;
public:
    // build
    explicit Tensor() = default;
    explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr); 
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr); 
    bool allocate_now(std::shared_ptr<base::DeviceAllocator> allocator, bool need_alloc = false);
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                                                                bool need_alloc, void* ptr);
    bool assign(std::shared_ptr<base::Buffer> buffer);
    
    // mem
    void to_cpu();
    void to_cuda(cudaStream_t stream = nullptr);
    Tensor clone() const;
    template <typename T>
    T* ptr();
    template <typename T>
    const T* ptr() const;

    // easy
    bool is_empty() const;
    base::DeviceType device_type() const;
    size_t byte_size() const;
    size_t size() const ;
    std::shared_ptr<base::Buffer> get_buffer() const;


};

// 模板函数的调用，源文件include的内容必须得看到完整实现，故不适合放在 tensor.cpp中

template <typename T>
T* Tensor::ptr(){
    if(!buffer_){
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr());// 强制类型转换
};

template <typename T>
const T* Tensor::ptr() const {
    if(!buffer_){
        return nullptr;
    }
    return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));// const_cast 主要用于移除或添加 const 属性
};

}
#endif