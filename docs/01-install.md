# Qwen3 推理

以下命令均在项目根目录 `KuiperLLama` 下执行。

## 1. 安装 Python 依赖

```shell
# pip install huggingface_hub torch transformers tqdm numpy
pip list | grep -E 'huggingface-hub|torch|transformers|tqdm|numpy'
```

## 2. 下载模型

模型会下载到当前目录下的 `Qwen/Qwen3-0.6B`：

```shell
# export HF_ENDPOINT=https://hf-mirror.com
# /home/l8w/miniconda3/envs/cp312/bin/python -m pip install huggingface-cli
# huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
#   --local-dir Qwen/Qwen3-0.6B \
#   --local-dir-use-symlinks False
ln -s /media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/Qwen/Qwen3-0.6B  /media/l8w/Linux118/PROJECTS/04-cpp/KuiperLLama/Qwen/Qwen3-0.6B 
```

## 3. 导出为 pth

```shell
python tools/export_qwen3/load.py \
  --model_name Qwen/Qwen3-0.6B \
  --output_file Qwen3-0.6B.pth
```

## 4. 导出为 bin

`write_bin.py` 会在 `tools/export_qwen3/` 下生成 `qwen0.6.bin`：

```shell
cd tools/export_qwen3
python write_bin.py \
  --model_name ../../Qwen/Qwen3-0.6B \
  --checkpoint ../../Qwen3-0.6B.pth
cd ../..
```

## 5. 编译项目

```shell
rm -rf build
cmake -S . -B build -DUSE_CPM=ON -DQWEN3_SUPPORT=ON
cmake --build build -j16
```

## 6. 运行推理

```shell
./build/demo/qwen3_infer \
  tools/export_qwen3/qwen0.6.bin \
  Qwen/Qwen3-0.6B/tokenizer_config.json \
  cuda
```

如果 CUDA 运行失败，先确认驱动可见：

```shell
nvidia-smi
```

如果 `nvidia-smi` 无法访问 NVIDIA driver，说明当前系统/容器/WSL 没有可用 GPU 驱动；修好驱动后再运行 CUDA 推理。也可以先用 CPU 做初始化和 tokenizer 冒烟测试：

```shell
./build/demo/qwen3_infer \
  tools/export_qwen3/qwen0.6.bin \
  Qwen/Qwen3-0.6B/tokenizer_config.json \
  cpu \
  0
```

## 7. 潜在问题


<details close>
<summary> ⚠ 展开/折叠 </summary>

```sh
# 1. 魔法问题
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1

# 2. 依赖目录问题
PROJECT_DIR="$(pwd)"

for dir in \
  "$PROJECT_DIR/build/_deps/gtest-src" \
  "$PROJECT_DIR/build/_deps/glog-src" \
  "$PROJECT_DIR/build/_deps/armadillo-src" \
  "$PROJECT_DIR/build/_deps/sentencepiece-src" \
  "$PROJECT_DIR/build/_deps/absl-src" \
  "$PROJECT_DIR/build/_deps/re2-src" \
  "$PROJECT_DIR/build/_deps/nlohmann_json-src"
do
  git config --global --add safe.directory "$dir"
done
```
</details>
