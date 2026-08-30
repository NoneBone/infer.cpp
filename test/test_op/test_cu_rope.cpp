#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../source/op/kernels/cpu/rope_kernel.h"
#include "../utils.cuh"
#include "base/buffer.h"
#include <random>
TEST(test_rope_cu, rope_nostream) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 256;
    int32_t head_size = 64;
    int32_t kv_dim = 128;
    int32_t pos = 3;
    tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    input_pos.index<int32_t>(0) = pos;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    tensor::Tensor input_q_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
    tensor::Tensor input_k_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
    constexpr int32_t max_seq_len = 128;
    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
    tensor::Tensor sin_cache_gpu(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cu);
    tensor::Tensor cos_cache_gpu(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cu);

    for (int i = 0; i < dim; ++i) {
    input_q_cpu.index<float>(i) = dist(mt);
    input_k_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor input_q_gpu = input_q_cpu.clone();
    tensor::Tensor input_k_gpu = input_k_cpu.clone();
    input_q_gpu.to_cuda(nullptr);
    input_k_gpu.to_cuda(nullptr);
    kernel::sin_cos_cache_calc_cpu(head_size, max_seq_len,
                                   sin_cache.ptr<float>(), cos_cache.ptr<float>());
    sin_cache_gpu = sin_cache.clone();
    cos_cache_gpu = cos_cache.clone();
    sin_cache_gpu.to_cuda(nullptr);
    cos_cache_gpu.to_cuda(nullptr);

    kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(
        dim, kv_dim, head_size, input_q_cpu, input_k_cpu, input_pos,
        sin_cache, cos_cache, nullptr);

    kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(
        dim, kv_dim, head_size, input_q_gpu, input_k_gpu, input_pos,
        sin_cache_gpu, cos_cache_gpu, nullptr);
    cudaDeviceSynchronize();

    input_q_gpu.to_cpu();
    input_k_gpu.to_cpu();
    for (int32_t i = 0; i < dim; ++i) {
    ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f)
        << "ik: " << i;
    ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f)
        << "iq: " << i;
    }
}

TEST(test_rope_cu, rope_stream1) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t dim = 512;
    int32_t head_size = 128;
    int32_t kv_dim = 32;
    int32_t pos = 4;
    tensor::Tensor input_pos(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    input_pos.index<int32_t>(0) = pos;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    tensor::Tensor input_q_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
    tensor::Tensor input_k_cpu(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
    constexpr int32_t max_seq_len = 128;
    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
    tensor::Tensor sin_cache_gpu(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cu);
    tensor::Tensor cos_cache_gpu(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cu);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int i = 0; i < dim; ++i) {
        input_q_cpu.index<float>(i) = dist(mt);
        input_k_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor input_q_gpu = input_q_cpu.clone();
    tensor::Tensor input_k_gpu = input_k_cpu.clone();
    input_q_gpu.to_cuda(nullptr);
    input_k_gpu.to_cuda(nullptr);

    kernel::sin_cos_cache_calc_cpu(head_size, max_seq_len,
                                   sin_cache.ptr<float>(), cos_cache.ptr<float>());
    sin_cache_gpu = sin_cache.clone();
    cos_cache_gpu = cos_cache.clone();
    sin_cache_gpu.to_cuda(nullptr);
    cos_cache_gpu.to_cuda(nullptr);

    kernel::get_rope_kernel(base::DeviceType::kDeviceCPU)(
        dim, kv_dim, head_size, input_q_cpu, input_k_cpu, input_pos,
        sin_cache, cos_cache, nullptr);

    kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(
        dim, kv_dim, head_size, input_q_gpu, input_k_gpu, input_pos, 
        sin_cache_gpu, cos_cache_gpu, nullptr);
    cudaDeviceSynchronize();

    input_q_gpu.to_cpu();
    input_k_gpu.to_cpu();
    for (int32_t i = 0; i < dim; ++i) {
    ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f)
        << "ik: " << i;
    ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f)
        << "iq: " << i;
    }
}
