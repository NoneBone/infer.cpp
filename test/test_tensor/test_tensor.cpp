#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "base/buffer.h"

#include "tensor/tensor.h"
#include "utils.cuh"

TEST(test_tensor, init_with_allocator) {
    using namespace base;
    auto alloc_cu = CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(DataType::kDataTypeFp16, size, true, alloc_cu);
    ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, init_empty) {
    using namespace base;
    auto alloc_cu = CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(DataType::kDataTypeFp16, size, false, alloc_cu);
    ASSERT_EQ(t1.is_empty(), true);
}

TEST(test_tensor, init_with_ptr) {
    using namespace base;
    int32_t size = 32 * 32;
    __half* ptr = new __half[size];
    ptr[0]=31.0f;

    tensor::Tensor t1(DataType::kDataTypeFp16, size, false, nullptr, ptr);
    ASSERT_EQ(t1.is_empty(), false);
    ASSERT_EQ(t1.ptr<__half>(), ptr);
    ASSERT_EQ(t1.ptr<__half>()[0], __float2half(31.0f));
}

TEST(test_tensor, to_cpu){
    using namespace base;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1(DataType::kDataTypeFp16, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1.is_empty(), false);
    set_value_cu_fp16(t1.ptr<__half>(), 32*32);

    t1.to_cpu();
    ASSERT_EQ(t1.device_type(), DeviceType::kDeviceCPU);
    for(int i=0; i<32*32;i++){
        ASSERT_EQ(t1.ptr<__half>()[i], __float2half(1.0f));
    }
}

TEST(test_tensor, init_with_assign){
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::kDataTypeFp16, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    int32_t size = 32 * 32;
    __half *ptr = new __half[size];
    for (int i = 0; i < size; ++i)
    {
        ptr[i] = __half(i);
    }
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(size * sizeof(__half),
                                                                          nullptr, ptr, true);
    buffer->set_device_type(base::DeviceType::kDeviceCPU);

    ASSERT_EQ(t1_cpu.assign(buffer), true);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    ASSERT_NE(t1_cpu.ptr<__half>(), nullptr);

    // verify
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(t1_cpu.ptr<__half>()[i], __half(i));
    }

    delete[] ptr;
}

TEST(test_tensor, clone){
    using namespace base;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1(DataType::kDataTypeFp16, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1.is_empty(), false);
    set_value_cu_fp16(t1.ptr<__half>(), 32*32);

    tensor::Tensor t2 = t1.clone();

    t2.to_cpu();
    ASSERT_EQ(t2.device_type(), DeviceType::kDeviceCPU);
    for(int i=0; i<32*32;i++){
        ASSERT_EQ(t2.ptr<__half>()[i], __float2half(1.0f));
    }
}

TEST(test_tensor, to_cuda){
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1(DataType::kDataTypeFp16, 32, 32, true, alloc);
    size_t size = 32*32;
    ASSERT_EQ(t1.is_empty(), false);
    for(int i=0; i<size;i++){
        t1.ptr<__half>()[i] = 1.0f;
    }

    t1.to_cuda();
    ASSERT_EQ(t1.device_type(), DeviceType::kDeviceCUDA);
    
    __half* ptr = new __half[size];
    cudaMemcpy(ptr, t1.ptr<__half>(), size*sizeof(__half), cudaMemcpyDeviceToHost);
    for(int i=0; i<32*32;i++){
        ASSERT_EQ(ptr[i], __float2half(1.0f));
    }

    delete[] ptr;
}
