#ifndef INFER_INCLUDE_BASE_H
#define INFER_INCLUDE_BASE_H
#include "glog/logging.h"
#include <cstring>
#include <string>

namespace base{
enum class DeviceType: uint8_t{
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
};
class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};
} // namespace base
#endif // INFER_INCLUDE_BASE_H