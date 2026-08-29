#include <cuda_runtime_api.h>
#include "../utils.cuh"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "base/buffer.h"

TEST(test_buffer, allocate){
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  {
    Buffer buffer(32, alloc);
    ASSERT_NE(buffer.ptr(), nullptr);
    LOG(INFO) << "start release";
  }
  LOG(INFO) << "finish release";
}

TEST(test_buffer, allocate2){
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<Buffer> buffer;
  {
    buffer = std::make_shared<Buffer>(32, alloc);
  }
  LOG(INFO) << "release";
}

TEST(test_buffer, use_ext){
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.ptr(), ptr);
  delete[] ptr;
}