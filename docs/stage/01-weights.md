# 阶段 1：safetensors 权重读取与 BF16 基础

## 实现

- 新增 `flux::SafetensorsLoader`，支持：
  - 单文件 `.safetensors`；
  - HuggingFace 分片 `.safetensors.index.json`；
  - 按 key 查询、shape/dtype/data offset 校验、按 key 加载；
  - CPU 加载，以及通过既有 `Tensor::to_cuda()` 的可选 CUDA 拷贝。
- 新增 `FluxWeightMap`：登记期望 key、dtype、shape，并在加载前检查缺失 key 或不匹配。
- `base::DataType` 与 `tensor::Tensor` 新增 BF16（2 bytes）支持。
- 本地 FLUX.1-dev 的 safetensors 为 BF16；loader 保留 BF16 位表示，BF16 是 diffusion 的主路径。
- 新增合成 safetensors 单文件/分片索引单测，覆盖 BF16/F32、shape、WeightMap 和按 key 读取。

## 已验证

临时验证程序已实际读取：

```
model/flux1dev/.../transformer/diffusion_pytorch_model.safetensors.index.json
```

结果：

- Transformer 索引：1160 个权重，其中 `transformer_blocks.` 前缀 608 个；
- `x_embedder.weight` 以 BF16 成功加载到 CPU；
- shape：`[3072, 64]`，BF16 byte size：393216。


```bash
export CPM_SOURCE_CACHE="$HOME/.cache/CPM"
cmake --build build -j8 --target test_llm
./build/test/test_llm --gtest_filter='test_flux_safetensors.*'
```
