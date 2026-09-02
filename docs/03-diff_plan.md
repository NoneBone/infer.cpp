# 工程迁移

## v0 方案

LLM 到 diffusion 模型的推理框架迁移，工程量较为庞大，这需要细致的拆解与流程设计。目前我概述了 2 两点，请你帮我完善后，在下方编写 v1 版方案。

### 详尽的差异分析

当前模型文件采用 qwen3 的 safetensor，转换为 pth 和 bin 后，输入推理引擎。

而我需要修改该仓库，以增添扩散模型推理，采用如下缓存的 hf 权重。
 ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/

目前已知的差异包括：
1. 基础算子缺失：flux transformer、文本编码器的算子
2. 计算模块缺失：CLIP/T5文本编码器
3. 需要采用 16 bit 的推理，所以全部算子需要重头写一遍，写法允许简单。

### 弥合差异路线

1. 权重读取部分

2. test_tensor 基础测试

3. 算子单元测试

4. flux.1 模型单图推理测试

5. 采用 diffuser 评估结果的两类误差，理论上应该 <1e-4


## v1 方案：分层迁移、逐级对齐、可回归验收

### 1. 目标与边界

v1 先建立一条可复现、可调试、可回归的 BF16 单图推理链路：

`safetensors/HuggingFace 权重 → 权重适配层 → CLIP/T5 → FLUX Transformer → VAE 解码 → 图片`

固定模型为 `black-forest-labs/FLUX.1-dev`，batch=1、单图输入和 BF16 主路径；先建立 PyTorch/diffusers 参考基线，再迁移到本仓库 C++/CUDA 后端。量化、批处理、动态 shape、LoRA、ControlNet、多卡和极致性能优化不纳入 v1。

### 2. 详尽的差异分析（补充）

1. **模型拓扑**：Qwen 是自回归 LLM；FLUX 是 latent diffusion Transformer，包含双流/单流 block、时间步与 guidance、RoPE、AdaLN、联合 attention 和多步 scheduler。
2. **输入输出**：LLM 输入 `input_ids`、输出 logits；FLUX 接收 prompt、随机 latent、尺寸、seed 和 guidance，输出 latent，再经 VAE 生成 RGB。
3. **权重组织**：HF 缓存包含分片 safetensors 及 `model_index.json`、tokenizer、scheduler、text encoder、transformer、vae 等组件，必须按组件和 key 映射。
4. **数值与布局**：统一 BF16/FP32 边界、NCHW/NHWC、`[B,S,H]` 与 `[S,B,H]`、latent/VAE scaling、RoPE 频率及 timestep/σ 定义。
5. **依赖与版本**：diffusers/transformers 仅作参考基线或权重解析工具；记录模型版本、权重 SHA256、依赖版本和许可证。

产物包括组件拓扑图、输入输出 tensor contract、HF key 映射表、算子清单、dtype/shape 表及风险清单。

### 3. 总体架构

建议新增并保持与现有 LLM 公共 tensor、设备、日志接口复用：

```text
diffusion/{io,text,flux,vae,scheduler,pipeline,reference,tests}
```

每个模块先提供 reference kernel，再替换为 CUDA kernel，并保留 reference 路径与 golden 数据。

### 4. 分阶段实施

每个阶段的工作报告，都需要保存，写为 docs/stage/0x-word.md，x 表示编号，word 需要用一个单词表示阶段关键词。
每个阶段新增的代码，需要尽量简单，并且贴近原始组织架构，不要引入新的风格。
你遇到了太复杂的任务，可以直接拆分成多个 step 来逐步实现。

#### 阶段 0：环境与基线冻结

固定模型目录（通过 `model/flux1dev/` 配置, 具体如下方软链接）、权重 SHA256、Python/PyTorch/diffusers/transformers/CUDA/GPU 版本。用 /opt/conda/envs/cp312/bin/python 官方 diffusers 固定 prompt、seed、尺寸、guidance、步数，保存 token ids、文本 hidden states、初始 latent、每个 denoise step latent 和 VAE 输出到 `tmp/golden/v1/`。
```
ln -s  ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/ /root/cys/PROJECT/05-github/infer.cpp/model/flux1d
ev
```

#### 阶段 1：权重读取与 tensor 基础设施

实现 safetensors 单文件/分片读取、索引解析、按 key 加载及 CPU/GPU 拷贝；建立 `WeightMap` 校验 key、shape、dtype、有限值。扩展 `test_tensor` 覆盖 BF16 张量、broadcast、reshape、transpose、concat/split、matmul、随机种子和设备拷贝；先完成各组件完整权重加载测试。

#### 阶段 2：CLIP/T5 文本编码器

严格复现 tokenizer 的 padding、EOS、attention mask、position ids 和最大长度。实现 embedding、LayerNorm/RMSNorm、attention、MLP、残差和最终投影；先 reference 对齐，再实现 BF16 CUDA kernel。固化输出 shape、dtype、scale、device contract，并保存每层 hidden state。

#### 阶段 3：FLUX Transformer

依次实现 time/guidance embedding、RoPE、AdaLN、double/single-stream block、QKV/attention、MLP 和输出头。每个 block 用随机输入与 PyTorch reference 比较，覆盖不同 sequence length、latent 分辨率和 mask。明确 image token 拼接/还原顺序、packed sequence offset 和 positional ids。

#### 阶段 4：scheduler、VAE 与 pipeline

实现与 diffusers 一致的 timestep/σ、Euler/Flow-Matching 更新、guidance 和 seed 初始化；实现 VAE decoder、latent scaling 及 `[-1,1]` 到 uint8 后处理。先串接文本编码、单步 Transformer、scheduler 更新和 VAE，再扩展完整 denoise loop，每步支持 dump。

#### 阶段 5：端到端验收与性能

固定配置完成单图生成并比较中间 tensor 和最终图像；正确性通过后再做 kernel fusion、显存复用、异步拷贝和 graph capture。记录总延迟、组件耗时、峰值显存和每步耗时，优化前后不得改变误差阈值。

### 5. 测试与误差验收

测试分四级：Tensor 级（输出、shape、stride、dtype、NaN/Inf）；算子级（随机输入与 PyTorch/reference 对比）；组件级（CLIP、T5、Transformer、VAE 关键层）；端到端级（固定 seed 的初始 latent、每步 latent 及最终图像）。

报告 `max_abs_error`、`mean_abs_error`，必要时报告相对误差、cosine similarity 和 RGB 像素误差。FP32/reference 目标为 `max_abs_error < 1e-4`；BF16 先以组件级 `1e-3` 作为诊断门槛，再依据官方 BF16 基线确定端到端阈值。放宽阈值必须记录原因和影响范围。

### 6. 里程碑与完成定义

- **M0**：环境、golden 基线和差异映射表。
- **M1**：权重读取、tensor 测试、全部组件权重加载。
- **M2**：CLIP/T5 对齐。
- **M3**：FLUX Transformer 单 block/全 block 对齐。
- **M4**：scheduler、VAE、pipeline 单图生成并回归通过。
- **M5**：性能报告、文档、示例命令和 CI 脚本。

完成条件：新环境可按文档配置模型并一键运行；测试可重复；失败可定位到权重、组件、block、denoise step 或 VAE；默认路径不破坏现有 Qwen/LLM；新增代码、模型版本、误差结果和限制均有记录。

### 7. 风险处理

- 显存不足时支持 CPU offload、按需加载和 attention slicing，但优先保证一致性。
- LayerNorm、softmax、归一化和 scheduler 累积可使用 FP32 累加，明确记录转换边界。
- key/shape 不匹配或模型版本不符时启动即失败并打印映射摘要。
- reference 与优化 kernel 并存，任何优化必须通过 golden 对比并提供回退开关。
