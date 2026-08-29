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
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){} //ISSUE

    virtual DeviceType device_type() const{
        return device_type_;
    }
    virtual void* allocate(size_t byte_size) const = 0;// ISSUE
    virtual void release(void* ptr) const = 0;// ISSUE

private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;// ISSUE
    void release(void* ptr) const override;// ISSUE
};

class CUDACPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CUDACPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;// ISSUE
    void release(void* ptr) const override;// ISSUE
private:
    int not_busy =0;
};
// factory entry
class CPUDeviceAllocatorFactory{
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance(){
        if(instance==nullptr){
           instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    };
private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};
} // namespace base
#endif