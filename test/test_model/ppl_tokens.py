"""向 C++ 困惑度测试输出与 test_ppl.py 完全相同的 token 序列（二进制）。"""

import array
import struct
import sys
from pathlib import Path

import pyarrow.ipc as ipc
from transformers import AutoTokenizer

from test_ppl import MAX_LENGTH, MODEL_PATH, SENTENCES


def write_ids(ids):
    sys.stdout.buffer.write(struct.pack("<I", len(ids)))
    sys.stdout.buffer.write(array.array("i", ids).tobytes())


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if sys.argv[1:] == ["sentences"]:
        sys.stdout.buffer.write(struct.pack("<I", len(SENTENCES)))
        for sentence in SENTENCES:
            encoded = sentence.encode("utf-8")
            sys.stdout.buffer.write(struct.pack("<I", len(encoded)))
            sys.stdout.buffer.write(encoded)
            write_ids(tokenizer(sentence, add_special_tokens=True,
                                truncation=True, max_length=MAX_LENGTH)["input_ids"])
    elif sys.argv[1:] == ["wikitext"]:
        arrow_file = Path("Qwen/wiki-text_test/wikitext-test.arrow")
        with arrow_file.open("rb") as source:
            lines = ipc.open_stream(source).read_all().column("text").to_pylist()
        text = "\n\n".join(lines)
        write_ids(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
    else:
        raise ValueError("usage: ppl_tokens.py {sentences|wikitext}")


if __name__ == "__main__":
    main()
