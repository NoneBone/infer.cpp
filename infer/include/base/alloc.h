#ifndef INFER_INCLUDE_ALLOC_H
#define INFER_INCLUDE_ALLOC_H
#include "glog/logging.h"
#include "base.h"
#include "memory"
#include "map"
namespace base{

class DeviceAllocator
{
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){}

    virtual DeviceType device_type() const{
        return device_type_;
    }
    virtual void* allocate(size_t byte_size) const = 0;// 虚函数，可以被子类重写。
    virtual void release(void* ptr) const = 0;// =0 表示纯虚函数，表示没有默认实现；该类为抽象类，不能直接创建对象

    virtual void memcpy(const void *src_ptr, void *dst_ptr, size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, 
                        void *stream = nullptr, bool need_sync = false) const;

private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;// 函数签名
    void release(void* ptr) const override;// override 表示这个函数必须重写基类虚函数
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};
// factory entry
class CPUDeviceAllocatorFactory{
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance(){
        if(instance==nullptr){
           instance = std::make_shared<CPUDeviceAllocator>();// 智能指针管理动态对象的生命周期
        }
        return instance;
    };
private:
    static std::shared_ptr<CPUDeviceAllocator> instance;     // instance 指针销毁时, CPUDeviceAllocator 对象自动释放
};

class CUDADeviceAllocatorFactory{
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance(){
        if(instance==nullptr){
           instance = std::make_shared<CUDADeviceAllocator>();// 智能指针管理动态对象的生命周期
        }
        return instance;
    };
private:
    static std::shared_ptr<CUDADeviceAllocator> instance;     // instance 指针销毁时, CPUDeviceAllocator 对象自动释放
};
} // namespace base
#endif