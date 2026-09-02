# 阶段 2：文本编码器必要算子单元测试

## 目标

为 FLUX.1-dev 的 CLIP 文本编码器和 T5 文本编码器建立 BF16 CPU reference 算子基线，后续 CUDA kernel 与完整模型层均以此作为数值对齐参考。

## 模型依据

本地模型配置：

- CLIP：hidden size=768，12 层，12 heads，使用 LayerNorm 与 QuickGELU；
- T5-XXL：d_model=4096，24 层，64 heads，使用 RMSNorm、GELU-New 与 gated FFN；
- 两个权重组件的 torch dtype 均为 BF16。

## 已实现

新增文件：

- kuiper/include/flux/text_kernels.h
- kuiper/source/flux/text_kernels.cpp
- test/test_flux/test_text_kernels.cpp

BF16 CPU reference 算子：

1. BF16 与 FP32 的显式转换；
2. LayerNorm：按最后一维归一化，支持 scale 和 bias；
3. RMSNorm：按最后一维 RMS 归一化，支持 scale；
4. QuickGELU：x / (1 + exp(-1.702 * x))；
5. GELU-New：T5 使用的 tanh 近似公式；
6. Stable Softmax：按最后一维先减最大值；
7. Gated-GELU：value * gelu_new(gate)。

所有归一化、指数、累积和在 FP32 进行；输入与输出保持 BF16。这是后续 BF16 CUDA 实现的数值边界。

## 单元测试

test_flux_text 包含：

- layer_norm_and_rms_norm：两个二维样本，对比 LayerNorm 与 RMSNorm 的预期值；
- activations_softmax_and_gated_gelu：QuickGELU、GELU-New、按行 Softmax 归一化和 T5 gated FFN。

```sh
依赖与配置指令
export CPM_SOURCE_CACHE="$HOME/.cache/CPM"

cmake -S . -B build \
  -DUSE_CPM=ON \
  -DQWEN3_SUPPORT=ON \
  -DCPM_SOURCE_CACHE="$CPM_SOURCE_CACHE"

预期执行命令：

cmake --build build -j16 --target test_llm
./build/test/test_llm --gtest_filter='test_flux_text.*:test_flux_safetensors.*'
```
## 当前验证状态

新增源码已用项目的 C++17/CUDA/glog/Armadillo 编译参数通过语法检查。

完整 test_llm 重建暂时受 CPM 依赖状态影响：重新配置后 build/_deps/glog-src 未完整恢复，缺少 glog/logging.h。此问题不属于文本算子实现；恢复 CPM 依赖后必须执行上述命令完成正式回归。

## 下一步

- 在阶段 2 后续实现 CLIP/T5 的 embedding、attention、MLP 与残差连接。
