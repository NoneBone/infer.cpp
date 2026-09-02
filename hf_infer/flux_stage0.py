#!/usr/bin/env python3
"""冻结 FLUX.1-dev 的本地环境、权重清单和 diffusers golden tensors。"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import FluxPipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "model" / "flux1dev"
DEFAULT_OUTPUT_DIR = ROOT / "tmp" / "golden" / "v1"
DEFAULT_PROMPT = "Beautiful picture of a wave breaking."

def resolve_model_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "model_index.json").is_file():
        return path
    candidates = sorted(p for p in (path / "snapshots").glob("*") if (p / "model_index.json").is_file())
    if len(candidates) != 1:
        raise RuntimeError(f"无法从 {path} 找到唯一的 FLUX 快照目录")
    return candidates[0]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"

def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor.detach().to("cpu").contiguous(), path)

def collect_manifest(model_dir: Path, output_dir: Path) -> None:
    files = [{"path": str(path.relative_to(model_dir)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(p for p in model_dir.rglob("*") if p.is_file())]
    import diffusers
    import safetensors
    import transformers
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "safetensors": safetensors.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvcc": command_output(["/usr/local/cuda/bin/nvcc", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
    }
    write_json(output_dir / "weights_manifest.json", {"model_dir": str(model_dir), "files": files})
    write_json(output_dir / "environment.json", environment)

def dump_golden(args: argparse.Namespace, model_dir: Path, output_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("阶段 0 golden 导出要求 CUDA GPU")
    device = torch.device("cuda")
    pipe = FluxPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    clip_tokens = pipe.tokenizer(args.prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids
    t5_tokens = pipe.tokenizer_2(args.prompt, padding="max_length", max_length=args.max_sequence_length,
                                 truncation=True, return_tensors="pt").input_ids
    save_tensor(output_dir / "clip_input_ids.pt", clip_tokens)
    save_tensor(output_dir / "t5_input_ids.pt", t5_tokens)
    with torch.inference_mode():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=args.prompt, prompt_2=args.prompt, device=device, num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length)
        save_tensor(output_dir / "prompt_embeds.pt", prompt_embeds)
        save_tensor(output_dir / "pooled_prompt_embeds.pt", pooled_prompt_embeds)
        save_tensor(output_dir / "text_ids.pt", text_ids)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        initial_latents, _ = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels // 4, args.height, args.width, prompt_embeds.dtype, device, generator)
        save_tensor(output_dir / "latent_initial.pt", initial_latents)
        def save_step(_pipe, step, timestep, callback_kwargs):
            save_tensor(output_dir / f"latent_step_{step:02d}.pt", callback_kwargs["latents"])
            (output_dir / f"step_{step:02d}.json").write_text(
                json.dumps({"step": step, "timestep": float(timestep.detach().cpu())}) + "\n")
            return callback_kwargs
        result = pipe(prompt=args.prompt, prompt_2=args.prompt, height=args.height, width=args.width,
                      guidance_scale=args.guidance_scale, num_inference_steps=args.steps,
                      max_sequence_length=args.max_sequence_length, latents=initial_latents, output_type="pt",
                      callback_on_step_end=save_step, callback_on_step_end_tensor_inputs=["latents"])
        save_tensor(output_dir / "vae_output.pt", result.images)
        pipe.image_processor.postprocess(result.images, output_type="pil")[0].save(output_dir / "image.png")
    write_json(output_dir / "config.json", {
        "model_dir": str(model_dir), "prompt": args.prompt, "seed": args.seed, "height": args.height,
        "width": args.width, "guidance_scale": args.guidance_scale, "steps": args.steps,
        "max_sequence_length": args.max_sequence_length, "dtype": "bfloat16"})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height 和 width 必须是 16 的倍数")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_dir(args.model_dir)
    collect_manifest(model_dir, output_dir)
    if not args.metadata_only:
        dump_golden(args, model_dir, output_dir)

if __name__ == "__main__":
    main()
