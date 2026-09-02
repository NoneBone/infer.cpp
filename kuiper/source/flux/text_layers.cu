#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>
#include "base/alloc.h"
#include "flux/text_layers.h"
#include "flux/text_kernels.h"

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
                                      const __nv_bfloat16* v, const __nv_bfloat16* bias, const __nv_bfloat16* mask, __nv_bfloat16* context, int seq,
                                      int hidden, int head_dim, bool causal, bool apply_scale) {
  int head = blockIdx.x / seq, row = blockIdx.x % seq, base = head * head_dim;
  extern __shared__ float scores[];
  if (threadIdx.x == 0) {
    float max_score = -INFINITY, scale = rsqrtf(static_cast<float>(head_dim));
    for (int col = 0; col < seq; ++col) {
      if ((causal && col > row) || (mask && __bfloat162float(mask[col]) == 0.f)) { scores[col] = -INFINITY; continue; }
      float dot = 0.f;
      for (int d = 0; d < head_dim; ++d)
        dot += __bfloat162float(q[row * hidden + base + d]) * __bfloat162float(k[col * hidden + base + d]);
      scores[col] = dot * (apply_scale ? scale : 1.f) + (bias ? __bfloat162float(bias[head * seq * seq + row * seq + col]) : 0.f);
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
  attention_bf16_kernel<<<seq * head_count, kThreads, sizeof(float) * seq>>>(ptr(q), ptr(k), ptr(v), attention_bias ? ptr(*attention_bias) : nullptr, nullptr, ptr(context), seq, hidden, head_dim, causal, true);
  check_cuda();
  linear_cuda(context, o_weight, nullptr, output);
}

__global__ void add_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=__float2bfloat16(__bfloat162float(a[i])+__bfloat162float(b[i])); }
__global__ void t5_bias_kernel(const __nv_bfloat16* weight, __nv_bfloat16* out, int buckets, int heads, int seq) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=heads*seq*seq)return; int h=i/(seq*seq), q=(i/seq)%seq, k=i%seq; int r=k-q, half=buckets/4, side=buckets/2, d=abs(r), off=r>0?side:0; int b=off+(d<half?d:min(side-1,half+int(logf(float(d)/half)/logf(128.f/half)*(side-half)))); out[i]=weight[b*heads+h]; }
void t5_encoder_block_cuda(const tensor::Tensor& input, const tensor::Tensor& n1w, const tensor::Tensor& qw, const tensor::Tensor& kw, const tensor::Tensor& vw, const tensor::Tensor& ow, const tensor::Tensor& rb, const tensor::Tensor& n2w, const tensor::Tensor& wi0, const tensor::Tensor& wi1, const tensor::Tensor& wo, int32_t heads, tensor::Tensor& output, const tensor::Tensor* attention_mask) {
  int seq=input.get_dim(0), hidden=input.get_dim(1), n=seq*hidden, ffn=wi0.get_dim(0);
  auto n1=cuda_temp({seq,hidden}), q=cuda_temp({seq,hidden}), k=cuda_temp({seq,hidden}), v=cuda_temp({seq,hidden}), ctx=cuda_temp({seq,hidden}), att=cuda_temp({seq,hidden}), r1=cuda_temp({seq,hidden}), n2=cuda_temp({seq,hidden}), a=cuda_temp({seq,ffn}), b=cuda_temp({seq,ffn}), g=cuda_temp({seq,ffn}), mlp=cuda_temp({seq,hidden}), bias=cuda_temp({heads,seq,seq});
  rms_norm(input,n1w,n1,1e-6f); linear_cuda(n1,qw,nullptr,q); linear_cuda(n1,kw,nullptr,k); linear_cuda(n1,vw,nullptr,v);
  t5_bias_kernel<<<(heads*seq*seq+kThreads-1)/kThreads,kThreads>>>(ptr(rb),ptr(bias),rb.get_dim(0),heads,seq); check_cuda();
  attention_bf16_kernel<<<seq*heads,kThreads,sizeof(float)*seq>>>(ptr(q),ptr(k),ptr(v),ptr(bias),attention_mask ? ptr(*attention_mask) : nullptr,ptr(ctx),seq,hidden,hidden/heads,false,false); check_cuda();
  linear_cuda(ctx,ow,nullptr,att); add_bf16_kernel<<<(n+kThreads-1)/kThreads,kThreads>>>(ptr(input),ptr(att),ptr(r1),n); rms_norm(r1,n2w,n2,1e-6f);
  linear_cuda(n2,wi0,nullptr,a); linear_cuda(n2,wi1,nullptr,b); gated_gelu(b,a,g); linear_cuda(g,wo,nullptr,mlp); add_bf16_kernel<<<(n+kThreads-1)/kThreads,kThreads>>>(ptr(r1),ptr(mlp),ptr(output),n); check_cuda();
}
}  // namespace flux
