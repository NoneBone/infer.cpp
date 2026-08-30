#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

#include "base/buffer.h"
#include "tensor/tensor.h"
#include "utils.cuh"
#include "../source/op/kernels/kernels_interface.h"

TEST(test_add_cu, add){
    using namespace base;
    int32_t size = 32*32;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor t2(DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor t3(DataType::kDataTypeFp32, size, true, alloc_cu);

    set_value_cu(t1.ptr<float>(), size, 2.0f);
    set_value_cu(t2.ptr<float>(), size, 3.0f);

    kernel::get_add_kernel(DeviceType::kDeviceCUDA)(t1, t2, t3, nullptr);
    cudaDeviceSynchronize();
    float* host = new float[size];
    cudaMemcpy(host, t3.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<size;i++){
        ASSERT_EQ(host[i], 5.0f);
    }
    delete[] host;
}

TEST(test_add_cu, add1_stream) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

    set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);// 与上方写法等价
    set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
    cudaDeviceSynchronize();
    float* output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(output[i], 5.f);
    }
    cudaStreamDestroy(stream);
    delete[] output;
}

TEST(test_add_cu, add_align1) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151 * 13;

    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

    set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
    set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);

    kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)( t1, t2, out, nullptr);
    cudaDeviceSynchronize();
    float* output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        ASSERT_NEAR(output[i], 5.4f, 0.1f);
    }

    delete[] output;
}