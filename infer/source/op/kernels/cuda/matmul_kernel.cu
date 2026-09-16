#include "matmul_kernel.cuh"
#include <cub/block/block_reduce.cuh>
namespace kernel
{
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp16(const __half* input, const __half* weight, __half* output, int M,
                                      int K) {
  // input:  [K]
  // weight: [M, K]，按 row-major 存储（MatmulLayer 约定为 [output, input]）
  // output: [M]
  // 每个线程负责一个输出元素，使用 FP32 做累加，最后转回 FP16。
  const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_idx >= M) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float input_value = __half2float(input[k]);
    const float weight_value = __half2float(weight[output_idx * K + k]);
    sum += input_value * weight_value;
  }

  output[output_idx] = __float2half(sum);
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  UNUSED(scale);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t M = weight.get_dim(0);  // output rows
  const int32_t K = weight.get_dim(1);  // input columns
  CHECK_EQ(K, input.get_dim(0));
  CHECK_EQ(output.get_dim(0), M);

  constexpr int threads = 128;
  const int blocks = (M + threads - 1) / threads;
  if (config && config->stream) {
    matmul_kernel_cu_fp16<128, 1><<<blocks, threads, 0, config->stream>>>(
        input.ptr<__half>(), weight.ptr<__half>(), const_cast<__half*>(output.ptr<__half>()), M, K);
  } else {
    matmul_kernel_cu_fp16<128, 1><<<blocks, threads>>>(input.ptr<__half>(), weight.ptr<__half>(),
                                              const_cast<__half*>(output.ptr<__half>()), M, K);
  }
}

} // namespace kernel
