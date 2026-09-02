# 阶段 0：环境与基线冻结

## 范围

本阶段固定 FLUX.1-dev 的本地权重入口、运行环境和 diffusers golden 生成配置，不修改现有 Qwen/LLM 推理路径。

## 实现

- 模型入口固定为仓库内的 `model/flux1dev` 软链接；脚本会解析其唯一的 HuggingFace snapshot。
- `hf_infer/flux_stage0.py` 生成 `tmp/golden/v1/`：
  - `weights_manifest.json`：模型快照中每个文件的相对路径、字节数和 SHA256；
  - `environment.json`：Python、PyTorch、diffusers、transformers、safetensors、CUDA、GPU、nvcc、驱动版本；
  - `config.json`：固定 prompt、seed、尺寸、guidance、步数、序列长度与 dtype（BF16）；
  - `clip_input_ids.pt`、`t5_input_ids.pt`；
  - `prompt_embeds.pt`、`pooled_prompt_embeds.pt`、`text_ids.pt`；
  - `latent_initial.pt`、每个 `latent_step_XX.pt`、`vae_output.pt` 及 `image.png`。

## 执行

```bash
/opt/conda/envs/cp312/bin/python hf_infer/flux_stage0.py
```

默认固定 512×512、seed=1234、guidance=3.5、4 steps。首次执行会对全部模型文件计算 SHA256，并完整运行一次 FLUX，因此耗时和显存占用较高。仅验证路径、环境及权重清单时：

```bash
/opt/conda/envs/cp312/bin/python hf_infer/flux_stage0.py --metadata-only
```

所有 golden 输出都在 `tmp/`，该目录已被 git 忽略；脚本和本报告进入版本管理。
