from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    return torch_module.bfloat16


def _call_pipeline(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    generator: Any,
    width: int = 0,
    height: int = 0,
) -> Any:
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "generator": generator,
    }
    if width > 0 and height > 0:
        kwargs["width"] = width
        kwargs["height"] = height
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    try:
        return pipe(**dict(kwargs, true_cfg_scale=guidance_scale)).images[0]
    except TypeError:
        pass
    try:
        return pipe(**dict(kwargs, guidance_scale=guidance_scale)).images[0]
    except TypeError:
        kwargs.pop("negative_prompt", None)
        return pipe(**dict(kwargs, guidance_scale=guidance_scale)).images[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-ref-image-plan", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--max-plans", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--background-width", type=int, default=1664)
    parser.add_argument("--background-height", type=int, default=928)
    args = parser.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    plan_path = Path(args.src_ref_image_plan)
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"image generation model dir does not exist: {model_dir}")
    plans = _load_jsonl(plan_path)
    if args.max_plans and args.max_plans > 0:
        plans = plans[: args.max_plans]
    if not plans:
        raise ValueError("src_ref image plan is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(torch, args.dtype),
        "trust_remote_code": True,
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    pipe = DiffusionPipeline.from_pretrained(str(model_dir), **load_kwargs)
    if not args.device_map:
        pipe = pipe.to(device)

    manifest: list[dict[str, Any]] = []
    for plan_index, plan in enumerate(plans, start=1):
        plan_id = str(plan.get("plan_id", "")).strip() or f"plan_{plan_index:04d}"
        candidate_dir = Path(str(plan.get("candidate_dir", "")).strip())
        if not candidate_dir:
            raise ValueError(f"plan {plan_id} is missing candidate_dir")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        prompts = [str(item).strip() for item in plan.get("image_prompts", []) if str(item).strip()]
        if not prompts:
            raise ValueError(f"plan {plan_id} is missing image_prompts")
        negative_prompt = str(plan.get("negative_prompt", "")).strip()
        role = str(plan.get("src_ref_role", "")).strip()
        width = int(plan.get("image_width") or 0)
        height = int(plan.get("image_height") or 0)
        if role == "background_reference" and (width <= 0 or height <= 0):
            width = int(args.background_width)
            height = int(args.background_height)
        num_candidates = max(1, int(plan.get("num_candidates", 1) or 1))
        generated: list[str] = []
        for candidate_index in range(1, num_candidates + 1):
            prompt = prompts[(candidate_index - 1) % len(prompts)]
            generator_device = "cpu" if args.device_map else device
            generator = torch.Generator(device=generator_device).manual_seed(args.seed + plan_index * 1000 + candidate_index)
            image = _call_pipeline(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                width=width,
                height=height,
            )
            image_path = candidate_dir / f"candidate_{candidate_index:03d}.png"
            image.save(image_path)
            generated.append(str(image_path))
        manifest.append(
            {
                "plan_id": plan_id,
                "candidate_dir": str(candidate_dir),
                "generated_images": generated,
                "image_width": width,
                "image_height": height,
                "model_dir": str(model_dir),
                "status": "generated",
            }
        )
    _write_jsonl(Path(args.output_manifest), manifest)
    print(json.dumps({"generated_plan_count": len(manifest), "output_manifest": args.output_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
