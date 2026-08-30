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
  }// 在这一行析构
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
}// 在这一行析构

TEST(test_buffer, use_ext){
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.ptr(), ptr);
  delete[] ptr;
}

TEST(test_buffer, cudaMemcpyDeviceToHost){
  using namespace base;
  auto alloc1 = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc2 = base::CUDADeviceAllocatorFactory::get_instance();
  int32_t size = 32;
  int32_t bytes_size = size * sizeof(float);
  float* ptr = new float[size];
  for(int i=0;i<size;i++){
    ptr[i] = i;
  }
  Buffer buffer(bytes_size, nullptr, ptr, true);
  buffer.set_device_type(DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer.is_external(), true);

  Buffer bufferCu(bytes_size, alloc2);
  bufferCu.copy_from(buffer);
  // verify
  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, bufferCu.ptr(), bytes_size, cudaMemcpyDeviceToHost);
  for(int i=0;i<size;i++){
    CHECK_EQ(ptr2[i], i);
  }

  delete[] ptr;
  delete[] ptr2;
}

TEST(test_buffer, mem_set_kernel){
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  int size = 32;
  int bytes_size = 32 * sizeof(float);
  Buffer buffer1(bytes_size, alloc_cu);
  Buffer buffer2(bytes_size, alloc_cu);
  ASSERT_EQ(buffer1.get_device_type(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(buffer2.get_device_type(), DeviceType::kDeviceCUDA);
  // set
  set_value_cu((float*) buffer1.ptr(), size, 1.0f);
  buffer2.copy_from(buffer1);
  // verify
  float* ptr = new float[32];
  cudaMemcpy(ptr, buffer1.ptr(), bytes_size, cudaMemcpyDeviceToHost);
  for(int i=0;i<size;i++){
    CHECK_EQ(ptr[i], 1.0f);
  }
}