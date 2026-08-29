# Qwen3 json解析


## 1. tokenizer_config.json 和 vocab.json

| Key                            | 含义                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| `add_bos_token`                | 编码文本时是否自动添加 BOS（序列开始）token。这里为 `false`。 |
| `add_prefix_space`             | 编码文本前是否自动添加空格。这里为 `false`，常见于 GPT-2 风格 tokenizer 的配置。 |
| `added_tokens_decoder`         | “额外 token”的反向映射表：token ID 映射到具体字符串及其属性。 |
| `additional_special_tokens`    | 额外特殊 token 列表。这些 token 通常不会被普通文本拆分，也会在解码时特殊处理。 |
| `bos_token`                    | BOS token。这里为 `null`，表示 Qwen3 没有单独定义 BOS token。 |
| `chat_template`                | Jinja 模板，用于把 `system`、`user`、`assistant`、`tool` 消息转换为模型实际接收的文本格式。 |
| `clean_up_tokenization_spaces` | 解码时是否自动清理多余空格。这里为 `false`，尽量保留原始空格。 |
| `eos_token`                    | EOS（序列结束）token。这里是 `<|im_end|>`。模型生成该 token 时通常表示回复结束。 |
| `errors`                       | 文本编码/解码遇到非法字符时的处理方式。`replace` 表示用替代字符替换。 |
| `model_max_length`             | tokenizer 支持的最大序列长度。这里是 `131072`，即 131072 个 token。实际限制还受模型和显存影响。 |
| `pad_token`                    | 填充 token。这里是 `<|endoftext|>`，用于把不同长度的输入补齐。 |
| `split_special_tokens`         | 是否将特殊 token 再拆分成普通字符。`false` 表示保持完整。    |
| `tokenizer_class`              | Transformers 加载 tokenizer 时使用的类。这里是 `Qwen2Tokenizer`，Qwen3 复用了 Qwen2 的 tokenizer 实现。 |
| `unk_token`                    | 未知 token。这里为 `null`，表示没有单独配置 `<unk>` token。  |

vocab.json: 词表映射，结构为 `{ "token文本": token_id }`。每个 token 对应整数 ID，用于文本与 token ID 的转换。

## 2. config.json

 模型结构与运行参数
 
| 编号 | 参数名                                  | 说明                            |
| ---- | --------------------------------------- | ------------------------------- |
| 1    | `architectures`                         | 模型类                          |
| 2    | `model_type`                            | 类型                            |
| 3    | `hidden_size`                           | 隐藏维度                        |
| 4    | `intermediate_size`                     | FFN 中间维度                    |
| 5    | `num_hidden_layers`                     | 层数                            |
| 6    | `num_attention_heads`                   | 注意力头数                      |
| 7    | `num_key_value_heads`                   | KV 头数（GQA）                  |
| 8    | `head_dim`                              | 头维度                          |
| 9    | `hidden_act`                            | 激活函数                        |
| 10   | `attention_bias`                        | 是否使用 bias                   |
| 11   | `attention_dropout`                     | 注意力 dropout                  |
| 12   | `rms_norm_eps`                          | RMSNorm epsilon                 |
| 13   | `initializer_range`                     | 初始化范围                      |
| 14   | `max_position_embeddings`               | 最大位置长度                    |
| 15   | `max_window_layers`                     | 窗口层配置                      |
| 16   | `sliding_window` / `use_sliding_window` | 滑动窗口配置                    |
| 17   | `rope_theta` / `rope_scaling`           | RoPE 配置                       |
| 18   | `bos_token_id` / `eos_token_id`         | 起止 token ID                   |
| 19   | `vocab_size`                            | 词表大小                        |
| 20   | `torch_dtype`                           | 权重类型                        |
| 21   | `tie_word_embeddings`                   | 是否共享 embedding 和输出层权重 |
| 22   | `use_cache`                             | 是否缓存 KV                     |
| 23   | `transformers_version`                  | Transformers 版本               |

## 3. generation_config.json

生成策略。

| 编号 | 参数名                 | 说明                      |
| ---- | ---------------------- | ------------------------- |
| 1    | `bos_token_id`         | 起始 ID                   |
| 2    | `eos_token_id`         | 结束 ID（可多个）         |
| 3    | `pad_token_id`         | 填充 ID                   |
| 4    | `do_sample`            | 是否采样                  |
| 5    | `temperature`          | 温度（越高越随机）        |
| 6    | `top_k`                | 保留概率最高的 K 个 token |
| 7    | `top_p`                | 保留累计概率达到 P 的候选 |
| 8    | `transformers_version` | 版本                      |


