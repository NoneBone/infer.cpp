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

    tensor::Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
    ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, init_empty) {
    using namespace base;
    auto alloc_cu = CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(DataType::kDataTypeFp32, size, false, alloc_cu);
    ASSERT_EQ(t1.is_empty(), true);
}

TEST(test_tensor, init_with_ptr) {
    using namespace base;
    int32_t size = 32 * 32;
    float* ptr = new float[size];
    ptr[0]=31.0f;

    tensor::Tensor t1(DataType::kDataTypeFp32, size, false, nullptr, ptr);
    ASSERT_EQ(t1.is_empty(), false);
    ASSERT_EQ(t1.ptr<float>(), ptr);
    ASSERT_EQ(t1.ptr<float>()[0], 31.0f);
}

TEST(test_tensor, to_cpu){
    using namespace base;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1.is_empty(), false);
    set_value_cu(t1.ptr<float>(), 32*32);

    t1.to_cpu();
    ASSERT_EQ(t1.device_type(), DeviceType::kDeviceCPU);
    for(int i=0; i<32*32;i++){
        ASSERT_EQ(t1.ptr<float>()[i], 1.0f);
    }
}

TEST(test_tensor, init_with_assign){
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    int32_t size = 32 * 32;
    float *ptr = new float[size];
    for (int i = 0; i < size; ++i)
    {
        ptr[i] = float(i);
    }
    std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(size * sizeof(float),
                                                                          nullptr, ptr, true);
    buffer->set_device_type(base::DeviceType::kDeviceCPU);

    ASSERT_EQ(t1_cpu.assign(buffer), true);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    ASSERT_NE(t1_cpu.ptr<float>(), nullptr);

    // verify
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(t1_cpu.ptr<float>()[i], float(i));
    }

    delete[] ptr;
}

TEST(test_tensor, clone){
    using namespace base;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1.is_empty(), false);
    set_value_cu(t1.ptr<float>(), 32*32);

    tensor::Tensor t2 = t1.clone();

    t2.to_cpu();
    ASSERT_EQ(t2.device_type(), DeviceType::kDeviceCPU);
    for(int i=0; i<32*32;i++){
        ASSERT_EQ(t2.ptr<float>()[i], 1.0f);
    }
}

TEST(test_tensor, to_cuda){
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1(DataType::kDataTypeFp32, 32, 32, true, alloc);
    size_t size = 32*32;
    ASSERT_EQ(t1.is_empty(), false);
    for(int i=0; i<size;i++){
        t1.ptr<float>()[i] = 1.0f;
    }

    t1.to_cuda();
    ASSERT_EQ(t1.device_type(), DeviceType::kDeviceCUDA);
    
    float* ptr = new float[size];
    cudaMemcpy(ptr, t1.ptr<float>(), size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<32*32;i++){
        ASSERT_EQ(ptr[i], 1.0f);
    }

    delete[] ptr;
}
