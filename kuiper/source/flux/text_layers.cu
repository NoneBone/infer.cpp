#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>
#include "base/alloc.h"
#include "flux/text_layers.h"

namespace flux {
namespace {
constexpr int kThreads = 256;
const __nv_bfloat16* ptr(const tensor::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.ptr<uint16_t>()); }
__nv_bfloat16* ptr(tensor::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.ptr<uint16_t>()); }
void check_cuda() { CHECK_EQ(cudaGetLastError(), cudaSuccess); }

// 每个线程计算一个输出元素，使用 FP32 累积、BF16 写回。
__global__ void linear_bf16_kernel(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                   const __nv_bfloat16* b, __nv_bfloat16* y, int rows,
                                   int in_dim, int out_dim) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * out_dim) return;
  int row = index / out_dim, out = index % out_dim;
  float sum = b ? __bfloat162float(b[out]) : 0.f;
  for (int col = 0; col < in_dim; ++col)
    sum += __bfloat162float(x[row * in_dim + col]) * __bfloat162float(w[out * in_dim + col]);
  y[index] = __float2bfloat16(sum);
}

// 正确性优先的 attention 基线：每个 block 处理一个 (head, query) 对。
__global__ void attention_bf16_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                      const __nv_bfloat16* v, const __nv_bfloat16* bias, __nv_bfloat16* context, int seq,
                                      int hidden, int head_dim, bool causal) {
  int head = blockIdx.x / seq, row = blockIdx.x % seq, base = head * head_dim;
  extern __shared__ float scores[];
  if (threadIdx.x == 0) {
    float max_score = -INFINITY, scale = rsqrtf(static_cast<float>(head_dim));
    for (int col = 0; col < seq; ++col) {
      if (causal && col > row) { scores[col] = -INFINITY; continue; }
      float dot = 0.f;
      for (int d = 0; d < head_dim; ++d)
        dot += __bfloat162float(q[row * hidden + base + d]) * __bfloat162float(k[col * hidden + base + d]);
      scores[col] = dot * scale + (bias ? __bfloat162float(bias[head * seq * seq + row * seq + col]) : 0.f);
      max_score = fmaxf(max_score, scores[col]);
    }
    float denom = 0.f;
    for (int col = 0; col < seq; ++col) { scores[col] = expf(scores[col] - max_score); denom += scores[col]; }
    for (int col = 0; col < seq; ++col) scores[col] /= denom;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float sum = 0.f;
    for (int col = 0; col < seq; ++col) sum += scores[col] * __bfloat162float(v[col * hidden + base + d]);
    context[row * hidden + base + d] = __float2bfloat16(sum);
  }
}
tensor::Tensor cuda_temp(const std::vector<int32_t>& dims) {
  return tensor::Tensor(base::DataType::kDataTypeBf16, dims, true, base::CUDADeviceAllocatorFactory::get_instance());
}
}  // namespace

void linear_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                 const tensor::Tensor* bias, tensor::Tensor& output) {
  int rows = input.get_dim(0), in_dim = input.get_dim(1), out_dim = weight.get_dim(0);
  int count = rows * out_dim;
  linear_bf16_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(ptr(input), ptr(weight), bias ? ptr(*bias) : nullptr, ptr(output), rows, in_dim, out_dim);
  check_cuda();
}
void self_attention_cuda(const tensor::Tensor& input, const tensor::Tensor& q_weight,
                         const tensor::Tensor& k_weight, const tensor::Tensor& v_weight,
                         const tensor::Tensor& o_weight, int32_t head_count, bool causal,
                         tensor::Tensor& output, const tensor::Tensor* attention_bias) {
  int seq = input.get_dim(0), hidden = input.get_dim(1), head_dim = hidden / head_count;
  auto q = cuda_temp({seq, hidden}), k = cuda_temp({seq, hidden});
  auto v = cuda_temp({seq, hidden}), context = cuda_temp({seq, hidden});
  linear_cuda(input, q_weight, nullptr, q); linear_cuda(input, k_weight, nullptr, k);
  linear_cuda(input, v_weight, nullptr, v);
  attention_bf16_kernel<<<seq * head_count, kThreads, sizeof(float) * seq>>>(ptr(q), ptr(k), ptr(v), attention_bias ? ptr(*attention_bias) : nullptr, ptr(context), seq, hidden, head_dim, causal);
  check_cuda();
  linear_cuda(context, o_weight, nullptr, output);
}
}  // namespace flux
