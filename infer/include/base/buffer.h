#ifndef INFER_INCLUDE_BUFFER_H
#define INFER_INCLUDE_BUFFER_H
#include "memory"
#include "base/alloc.h"
namespace base {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer>{
private:
    size_t byte_size_;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;
    void* ptr_ = nullptr;
    bool use_external_ = false;
public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_external = false);
    virtual ~Buffer();

    bool allocate(); // TODO
    void copy_from(const Buffer& buffer) const;// 必须传入一个 Buffer 对象,const 表示只能读取a的成员，不能修改
    void copy_from(const Buffer* buffer) const;// 传入的地址，可以是nullptr

    size_t byte_size() const;
    DeviceType get_device_type() const;
    void set_device_type(DeviceType device_type);
    std::shared_ptr<DeviceAllocator> get_allocator() const;
    void* ptr();
    const void* ptr() const;
    bool is_external() const;
};
} // namespace base
#endif