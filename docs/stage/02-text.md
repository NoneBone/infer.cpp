# 阶段 2：CLIP/T5 文本编码器

## 目标

建立 FLUX.1-dev 文本编码器的 BF16 reference 基础层。实现优先保证与 HuggingFace 权重布局和数学定义一致，性能优化留给后续 CUDA 专项阶段。

## 权重映射

CLIP 使用 text_encoder：

- text_model.embeddings.token_embedding.weight
- text_model.embeddings.position_embedding.weight
- text_model.encoder.layers.N.layer_norm1/2
- text_model.encoder.layers.N.self_attn.q_proj/k_proj/v_proj/out_proj
- text_model.encoder.layers.N.mlp.fc1/fc2

T5 使用 text_encoder_2：

- shared.weight
- encoder.block.N.layer.0.layer_norm
- encoder.block.N.layer.0.SelfAttention.q/k/v/o
- encoder.block.N.layer.1.layer_norm
- encoder.block.N.layer.1.DenseReluDense.wi_0/wi_1/wo

## 已实现

新增 text_layers.h、text_layers.cpp 和 test_text_layers.cpp。

CPU BF16 reference 层：

1. Embedding：int32 token id 查表，支持可选 position embedding；
2. Linear：权重布局为 [out_features, in_features]，FP32 累积、BF16 输出；
3. Residual Add：两个同 shape BF16 tensor 相加；
4. Multi-Head Self-Attention：Q/K/V/O 投影、缩放点积、可选 causal mask、softmax、head 拼接；
5. CLIP MLP：fc1 -> QuickGELU -> fc2；
6. T5 Gated MLP：wi_0、wi_1、GELU-New gate、wo。

BF16 CUDA 基础算子已提供 LayerNorm、RMSNorm、QuickGELU、GELU-New、Softmax、Gated-GELU；text_flux_text 的 CUDA/CPU 对比测试已通过。

## 单元测试

test_flux_layers 覆盖：

- embedding、position embedding、linear 和 residual；
- 双 token、单 head self-attention；
- CLIP MLP 和 T5 gated MLP。

执行：

    cmake --build build -j8 --target test_llm
    ./build/test/test_llm --gtest_filter='test_flux_layers.*:test_flux_text.*'

## 验证结果

已执行 text_flux_layers 和 text_flux_text，共 5 项测试全部通过；其中 cuda_matches_cpu 验证 LayerNorm 和 Softmax 的 BF16 CPU/CUDA 输出一致。

## 限制与下一步

已增加 CUDA Linear 与 Self-Attention reference kernel：前者按输出元素 FP32 累积，后者按 (head, query) block 计算；两者均保持 BF16 存储。

## 掩码与 golden 对齐

- CLIP：`clip_causal_mask` 产生 `[seq, seq]` 的下三角允许矩阵（允许位置为 0，未来位置为 `-inf`），attention 的 `causal=true` 采用等价逻辑。
- T5：`t5_relative_position_bucket` 按 T5 的 bidirectional bucket 规则映射相对位置；`t5_relative_position_bias` 从 HF 权重 `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight`（`[num_buckets, heads]`）生成 `[heads, seq, seq]` bias，可作为 attention 的 `attention_bias` 参数叠加到 softmax 前 score。
- 真实 golden：`hf_infer/flux_stage0.py` 现会额外导出 `clip_block_00_input/output.pt`（`[1, 77, 768]`）和 `t5_block_00_input/output.pt`（`[1, 512, 4096]`），均来自本地 FLUX 权重上的 diffusers 推理、dtype 为 BF16。`prompt_embeds.pt` 为 `[1, 512, 4096]`，`pooled_prompt_embeds.pt` 为 `[1, 768]`；它们分别构成 block 级和端到端对齐目标。

新增单测覆盖 T5 相对 bucket/bias 和 CLIP 未来 token mask；端到端 prompt embedding 的逐元素比对需在完整 CLIP/T5 block 编排后接入，因为目前尚未实现完整层堆叠与 `.pt` 读取器。所有输入输出均以 BF16 存储，归一化、softmax 和 GEMM 累积保持 FP32。


## CLIP block reference

已加入 `clip_encoder_block` 的 CPU BF16 reference：pre-LayerNorm、包含 q/k/v/out bias 的因果 self-attention、两次残差连接及 QuickGELU MLP。已由 safetensors 权重加载器读入 `text_model.encoder.layers.0.*`，并在 `test_flux_layers.clip_block_00_matches_real_golden` 对比 `clip_block_00_output.pt` 转换得到的 safetensors golden。测试于 2026-09-02 通过，CPU reference 耗时约 17.8 秒，逐元素最大绝对误差阈值为 0.20（BF16 容差）。


## T5 CUDA block（进行中）

已实现 CUDA-only 的 RMSNorm、Q/K/V/O Linear、相对位置 bias、无 `1/sqrt(d)` 缩放 attention、残差与 gated-GELU MLP，并接入 `text_encoder_2/model.safetensors.index.json` 的第 0 block 真实权重和 `t5_block_00_golden.safetensors`。`test_flux_layers.t5_block_00_cuda_matches_real_golden` 已实际执行，但当前最大绝对误差为 198.125，尚未达到 BF16 对齐阈值，不能作为通过项；后续需以 Transformers 的 attention mask 与 position-bias 中间值为依据继续定位。


## T5 CUDA relative-bias 修复与验证（2026-09-02）

### 修复内容

CUDA 的 `t5_bias_kernel` 已按 Transformers T5 的双向 bucket 规则生成 position bias。此前 kernel 虽计算了正相对位置的 bucket 偏移 `off`，但最终索引遗漏该偏移，使正向位置错误读取负向 bucket。修复为将 `off` 加入最终 bucket 索引。

T5 CUDA block 保持全 CUDA 算子路径：RMSNorm、Q/K/V/O Linear、relative position bias、无 `1/sqrt(head_dim)` 缩放的 attention、两次 residual add 与 gated-GELU MLP。CPU 只用于 safetensors/golden 文件读取及最终结果回传比较，不参与推理算子。

### 真实权重与 golden

测试从 `text_encoder_2/model.safetensors.index.json` 加载 `encoder.block.0` 的 RMSNorm、Q/K/V/O、relative attention bias、FFN RMSNorm 和 wi_0/wi_1/wo 分片权重；输入/输出使用 `tmp/golden/v1/t5_block_00_golden.safetensors`，其来源为 diffusers 导出的 `t5_block_00_input/output.pt`。

### 可运行指令

```bash
cmake --build build -j1 --target test_llm
./build/test/test_llm --gtest_filter=test_flux_layers.t5_block_00_cuda_matches_real_golden
```

### 验证结果

修复前最大绝对误差为 `69.25`；修复 relative-bias 正向 bucket 偏移后降为 `1.0`。主语义错误已修复，但该值尚未满足当前 `0.30` 的绝对误差门限。剩余差异需结合 mean/relative error 评估 BF16 逐元素 CUDA 累积与 PyTorch 矩阵核的舍入差异，不能通过简单放宽最大绝对误差门限掩盖。
