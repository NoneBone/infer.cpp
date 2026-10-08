"""使用本地 Qwen3-4B FP16 模型计算三句话及 WikiText-2 test 的困惑度。"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path("/media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B")
MAX_LENGTH = 128
WIKITEXT_BLOCK_SIZE = 1024
SENTENCES = (
    "人工智能正在改变人们获取和处理信息的方式。",
    "今天天气很好，我们一起去公园散步。",
    "深度学习模型通过大量数据学习复杂的特征表示。",
)

@dataclass(frozen=True)
class PerplexityResult:
    predicted_tokens: int
    total_nll: float

    @property
    def mean_nll(self) -> float:
        return self.total_nll / self.predicted_tokens

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll)


def calculate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> PerplexityResult:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if input_ids.size(1) < 2:
        raise ValueError("文本编码后至少需要两个 token 才能计算困惑度")

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    # logits[:, t] 预测 input_ids[:, t + 1]。交叉熵使用 FP32，降低
    # FP16 softmax 带来的数值误差。ISSUE 为什么要 shift？
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    total_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    )

    return PerplexityResult(
        predicted_tokens=shift_labels.numel(),
        total_nll=total_nll.item(),
    )


def calculate_wikitext_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> PerplexityResult:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    if len(token_ids) < 2:
        raise ValueError("WikiText-2 test 编码后至少需要两个 token")

    total_nll = 0.0
    total_tokens = 0
    # 每块与前一块重叠一个 token，使所有后续 token 都恰好被预测一次。
    for start in range(0, len(token_ids) - 1, WIKITEXT_BLOCK_SIZE - 1):
        block = torch.tensor(
            token_ids[start : start + WIKITEXT_BLOCK_SIZE],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        with torch.inference_mode():
            logits = model(input_ids=block, use_cache=False).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
                block[:, 1:].reshape(-1),
                reduction="sum",
            )
        total_nll += loss.item()
        total_tokens += block.size(1) - 1

    return PerplexityResult(predicted_tokens=total_tokens, total_nll=total_nll)


def main() -> None:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("该测试需要可用的 CUDA 设备")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    model.eval()

    results: list[PerplexityResult] = []
    print("index,tokens,mean_nll,perplexity,text")
    for index, sentence in enumerate(SENTENCES, start=1):
        result = calculate_perplexity(model, tokenizer, sentence, device)
        results.append(result)
        print(
            f'{index},{result.predicted_tokens},{result.mean_nll:.6f},'
            f'{result.perplexity:.6f},"{sentence}"'
        )

    total_tokens = sum(result.predicted_tokens for result in results)
    total_nll = sum(result.total_nll for result in results)
    corpus_mean_nll = total_nll / total_tokens
    corpus_perplexity = math.exp(corpus_mean_nll)
    print(f"overall,{total_tokens},{corpus_mean_nll:.6f},{corpus_perplexity:.6f},")

    wikitext_result = calculate_wikitext_perplexity(model, tokenizer, device)
    print(
        f"wikitext-2-test,{wikitext_result.predicted_tokens},"
        f"{wikitext_result.mean_nll:.6f},{wikitext_result.perplexity:.6f},"
    )


if __name__ == "__main__":
    main()
