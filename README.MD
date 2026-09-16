# infer.cu

## start

### 1. install

```shell
rm -rf build
cmake -S . -B build -DUSE_CPM=ON -DQWEN3_SUPPORT=ON
cmake --build build -j16
```

### 2. infer

```sh
python tools/export_qwen3/export_qwen3-fp16.py \
		Qwen3-4B-fp16.bin \
		--dtype fp16 --hf /media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B

./build/demo/qwen3_infer \
    Qwen3-4B-fp16.bin \
    /media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B/tokenizer.json
				 # \
    # --stream --max-tokens 1024 --interactive
		# --benchmark
		# --benchmark-decode-tokens 1024
		# --attnOpt true
		# --gemmOpt true 
		# --rmsNorm true
		# --ropeOpt true
```

### 3. exp
```sh

```
