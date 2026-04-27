from __future__ import annotations

import unittest
from pathlib import Path


class ScriptTests(unittest.TestCase):
    def test_omni_detective_script_uses_own_repo_root(self) -> None:
        script = Path("scripts/run_omni_detective_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("REPO_ROOT=", script)
        self.assertIn('cd "$REPO_ROOT"', script)
        self.assertIn('export PYTHONPATH="$REPO_ROOT', script)
        self.assertNotIn("cd /data02/usr/wangqihao/Demo/test/cvr", script)
        self.assertNotIn("PYTHONPATH=/data02/usr/wangqihao/Demo/test/cvr", script)

    def test_omni_detective_script_has_gpu_resource_policy(self) -> None:
        script = Path("scripts/run_omni_detective_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("MAX_GPUS", script)
        self.assertIn("GPU_IDS", script)
        self.assertIn("MODEL_STAGE", script)
        self.assertIn("one Omni model per run", script)
        self.assertIn("refusing to run with GPU_COUNT", script)

    def test_omni_detective_script_accepts_run_root_cli_override(self) -> None:
        script = Path("scripts/run_omni_detective_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("--run-root", script)
        self.assertIn('RUN_ROOT="$2"', script)
        self.assertNotIn("omni_detective_pilot_20260422", script)
        self.assertIn("omni_detective_pilot", script)

    def test_video_edit_env_script_is_read_only_and_checks_wan_layout(self) -> None:
        script = Path("scripts/check_video_edit_env.sh").read_text(encoding="utf-8")

        self.assertIn("03_audio_vlm2vec_backbone", script)
        self.assertIn("Wan2.1-VACE-1.3B", script)
        self.assertIn("Wan2.1-VACE-14B", script)
        self.assertIn("huggingface_hub", script)
        self.assertIn("nvidia-smi", script)
        self.assertIn("never downloads or runs generation", script)
        self.assertNotIn("snapshot_download", script)
        self.assertNotIn("python -m app.composed_data", script)

    def test_wan_vace_download_script_targets_expected_models(self) -> None:
        script = Path("scripts/download_wan_vace_models.sh").read_text(encoding="utf-8")

        self.assertIn("Wan-AI/Wan2.1-VACE-1.3B", script)
        self.assertIn("Wan-AI/Wan2.1-VACE-14B", script)
        self.assertIn("Wan-Video/Wan2.1.git", script)
        self.assertIn("--model-size 1.3B|14B|both", script)
        self.assertIn("--provider modelscope|hf", script)
        self.assertIn("DOWNLOAD_PROVIDER=${DOWNLOAD_PROVIDER:-modelscope}", script)
        self.assertIn("modelscope download", script)
        self.assertIn("03_audio_vlm2vec_backbone", script)
        self.assertIn("snapshot_download", script)

    def test_synthetic_validation_script_runs_known_pair_validation(self) -> None:
        script = Path("scripts/run_synthetic_known_pairs_validation.sh").read_text(encoding="utf-8")

        self.assertIn("--run-root", script)
        self.assertIn("--known-pairs", script)
        self.assertIn("--clip-annotations", script)
        self.assertIn("validate-known-pairs", script)
        self.assertIn("judged_synthetic_pair_proposals.jsonl", script)
        self.assertIn("accepted_synthetic_pairs.jsonl", script)
        self.assertIn("synthetic_pilot_review.md", script)
        self.assertIn("one Omni model per run", script)
        self.assertIn("refusing to run with GPU_COUNT", script)

    def test_dual_route_download_script_targets_visual_and_audio_models(self) -> None:
        script = Path("scripts/download_dual_route_models.sh").read_text(encoding="utf-8")

        self.assertIn("LTX-2", script)
        self.assertIn("LTX-Video", script)
        self.assertIn("FoleyCrafter", script)
        self.assertIn("Frieren-V2A", script)
        self.assertIn("modelscope download", script)
        self.assertIn("omni_src", script)

    def test_mask_edit_download_script_targets_grounded_sam_models(self) -> None:
        script = Path("scripts/download_mask_edit_models.sh").read_text(encoding="utf-8")

        self.assertIn("Grounded-SAM-2", script)
        self.assertIn("GroundingDINO", script)
        self.assertIn("SAM2.1", script)
        self.assertIn("Florence-2", script)
        self.assertIn("facebook/sam2.1-hiera-large", script)
        self.assertIn("modelscope download", script)

    def test_deterministic_audio_smoke_preserves_video_with_ffmpeg(self) -> None:
        script = Path("scripts/run_deterministic_audio_synthetic_smoke.sh").read_text(encoding="utf-8")

        self.assertIn("deterministic audio only", script)
        self.assertIn("MAX_AUDIO_SAMPLES", script)
        self.assertIn("amix=inputs=2", script)
        self.assertIn("dropout_transition=0[a]", script)
        self.assertNotIn("normalize=0", script)
        self.assertIn("overlay_reference_audio", script)
        self.assertIn("$RUN_ROOT/logs", script)
        self.assertIn("wind noise", script)
        self.assertIn("high-pitched beep", script)
        self.assertIn("-map 0:v:0", script)
        self.assertIn("ffmpeg", script)
        self.assertIn("synthetic_candidate_pairs.jsonl", script)
        self.assertIn("deterministic_overlay", script)

    def test_dual_route_validation_script_uses_synthetic_validation_entrypoint(self) -> None:
        script = Path("scripts/run_synthetic_dual_route_validation.sh").read_text(encoding="utf-8")

        self.assertIn("one Omni model per run", script)
        self.assertIn("--known-pairs", script)
        self.assertIn("validate-known-pairs", script)
        self.assertIn("accepted_synthetic_pairs.jsonl", script)
        self.assertIn("synthetic_pilot_review.md", script)
        self.assertIn("/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct", script)

    def test_vace_visual_synthetic_smoke_uses_plan_and_remuxes_audio(self) -> None:
        script = Path("scripts/run_vace_visual_synthetic_smoke.sh").read_text(encoding="utf-8")

        self.assertIn("video_edit_plan.jsonl", script)
        self.assertIn("vace_controlled", script)
        self.assertIn("Wan2.1-VACE-1.3B", script)
        self.assertIn("vace-14B", script)
        self.assertIn("VACE_TASK", script)
        self.assertIn("--vace-task", script)
        self.assertIn("ULYSSES_SIZE", script)
        self.assertIn("RING_SIZE", script)
        self.assertIn("--ulysses-size", script)
        self.assertIn("--ring-size", script)
        self.assertIn("--ring_size", script)
        self.assertIn('"model": wan_ckpt.name', script)
        self.assertNotIn('"model": "Wan2.1-VACE-1.3B"', script)
        self.assertIn("CONDA_ENV=${CONDA_ENV:-wan_vace}", script)
        self.assertIn("--conda-env", script)
        self.assertIn("flash_attn", script)
        self.assertIn("--max-gpus", script)
        self.assertIn('MAX_GPUS="$2"', script)
        self.assertIn("ALLOW_CPU_OFFLOAD=${ALLOW_CPU_OFFLOAD:-0}", script)
        self.assertIn("refusing CPU offload", script)
        self.assertIn("refusing CPU text encoder", script)
        self.assertIn("--src_video", script)
        self.assertIn("--prompt", script)
        self.assertIn("--task \"$VACE_TASK\"", script)
        self.assertIn("--offload_model \"$OFFLOAD_MODEL\"", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=\"$GPU_IDS\"", script)
        self.assertIn("torchrun --nproc_per_node", script)
        self.assertIn("-map 0:v:0 -map 1:a?", script)
        self.assertIn("audio_copied_from_reference", script)
        self.assertIn("--mask-manifest", script)
        self.assertIn("--src_mask", script)
        self.assertIn("--src_ref_images", script)
        self.assertIn("SRC_REF_IMAGES", script)
        self.assertIn("--src-ref-selection", script)
        self.assertIn("src_mask", script)
        self.assertIn("synthetic_visual_candidate_pairs.jsonl", script)
        self.assertIn("synthetic_visual_target_manifest.jsonl", script)

    def test_vace_visual_batch_script_runs_plan_ids_and_collects_pairs(self) -> None:
        script = Path("scripts/run_vace_visual_batch_from_plan.sh").read_text(encoding="utf-8")

        self.assertIn("PLAN_IDS", script)
        self.assertIn("TOP_K", script)
        self.assertIn("run_vace_visual_synthetic_smoke.sh", script)
        self.assertIn("mapfile -t SELECTED_PLAN_IDS", script)
        self.assertIn("< /dev/null", script)
        self.assertIn("MASK_MANIFEST", script)
        self.assertIn("--mask-manifest", script)
        self.assertIn("SRC_REF_SELECTION", script)
        self.assertIn("--src-ref-selection", script)
        self.assertIn("synthetic_visual_candidate_pairs.jsonl", script)
        self.assertIn("synthetic_visual_target_manifest.jsonl", script)
        self.assertIn("batch_generation_report.md", script)
        self.assertIn("This script does not start Omni", script)

    def test_vace_masked_visual_batch_script_requires_mask_manifest(self) -> None:
        script = Path("scripts/run_vace_masked_visual_batch_from_plan.sh").read_text(encoding="utf-8")

        self.assertIn("video_mask_manifest.jsonl", script)
        self.assertIn("missing mask manifest", script)
        self.assertIn("run_vace_visual_batch_from_plan.sh", script)
        self.assertIn("--mask-manifest", script)
        self.assertIn("--src-ref-selection", script)
        self.assertIn("This script does not start Omni", script)

    def test_grounded_sam2_mask_script_generates_manifest(self) -> None:
        script = Path("scripts/run_grounded_sam2_video_masks.sh").read_text(encoding="utf-8")
        helper = Path("scripts/generate_grounded_sam2_video_masks.py").read_text(encoding="utf-8")

        self.assertIn("video_mask_plan.jsonl", script)
        self.assertIn("video_mask_manifest.generated.jsonl", script)
        self.assertIn("Grounded-SAM-2", script)
        self.assertIn("CUDA_VISIBLE_DEVICES", script)
        self.assertIn("generate_grounded_sam2_video_masks.py", script)
        self.assertIn("build_sam2_video_predictor", helper)
        self.assertIn("GroundingDINO found no box", helper)
        self.assertIn("Florence-2 found no box", helper)
        self.assertIn("--grounder", script)
        self.assertIn("--florence-model", script)
        self.assertIn("florence2", helper)
        self.assertNotIn("torch_dtype=torch.float16", helper)
        self.assertIn("mask_temporal_stability", helper)
        self.assertIn("edit_background_inverse_subject", helper)

    def test_prepare_grounded_sam2_env_script_installs_expected_packages(self) -> None:
        script = Path("scripts/prepare_grounded_sam2_env.sh").read_text(encoding="utf-8")

        self.assertIn("grounded_sam2", script)
        self.assertIn("conda create", script)
        self.assertIn("download.pytorch.org/whl/cu121", script)
        self.assertIn("SAM2.1/code", script)
        self.assertIn("GroundingDINO/code", script)
        self.assertIn("torch.cuda.is_available", script)

    def test_repair_groundingdino_env_script_uses_no_build_isolation(self) -> None:
        script = Path("scripts/repair_groundingdino_env.sh").read_text(encoding="utf-8")

        self.assertIn("grounded_sam2", script)
        self.assertIn("GROUNDINGDINO_USE_CUDA", script)
        self.assertIn("FORCE_CUDA", script)
        self.assertIn("--no-build-isolation -e .", script)
        self.assertIn("groundingdino.groundingdino.util.inference", script)

    def test_repair_florence2_env_script_pins_transformers_compat(self) -> None:
        script = Path("scripts/repair_florence2_env.sh").read_text(encoding="utf-8")

        self.assertIn("grounded_sam2", script)
        self.assertIn("transformers>=4.45,<4.50", script)
        self.assertIn("trust_remote_code=True", script)
        self.assertIn("export FLORENCE_MODEL", script)
        self.assertIn("AutoProcessor.from_pretrained", script)
        self.assertIn("AutoModelForCausalLM.from_pretrained", script)

    def test_manual_review_bundle_script_calls_review_bundle_command(self) -> None:
        script = Path("scripts/build_synthetic_manual_review_bundle.sh").read_text(encoding="utf-8")

        self.assertIn("build-review-bundle", script)
        self.assertIn("--pairs-path", script)
        self.assertIn("--output-dir", script)
        self.assertIn("reference.mp4, target.mp4, review.md, metadata.json", script)

    def test_masked_vace_pipeline_queue_keeps_omni_and_splits_gpus(self) -> None:
        script = Path("scripts/run_masked_vace_pipeline_queue.sh").read_text(encoding="utf-8")

        self.assertIn("never starts or stops Omni", script)
        self.assertIn("MASK_GPU_IDS=${MASK_GPU_IDS:-6}", script)
        self.assertIn("VACE_GPU_IDS=${VACE_GPU_IDS:-2,3,4,5}", script)
        self.assertIn("plan-stable-omni-clips", script)
        self.assertIn("cache-reference-understandings", script)
        self.assertIn("plan-src-ref-images", script)
        self.assertIn("run_src_ref_image_generation_from_plan.sh", script)
        self.assertIn("select-src-ref-images", script)
        self.assertIn("PLANNING_MODE=${PLANNING_MODE:-production}", script)
        self.assertIn("plan-video-edits", script)
        self.assertIn("--planning-mode \"$PLANNING_MODE\"", script)
        self.assertIn("--planner-cache-path \"$VIDEO_EDIT_PLANNER_CACHE\"", script)
        self.assertIn("plan-video-masks", script)
        self.assertIn("run_grounded_sam2_video_masks.sh", script)
        self.assertIn("run_vace_masked_visual_batch_from_plan.sh", script)
        self.assertIn("detective-annotate-clips", script)
        self.assertIn("validate-known-pairs", script)
        self.assertIn("build_synthetic_manual_review_bundle.sh", script)
        self.assertIn("florence2", script)
        self.assertIn("vace-14B", script)

    def test_download_image_generation_models_script_uses_modelscope(self) -> None:
        script = Path("scripts/download_image_generation_models.sh").read_text(encoding="utf-8")

        self.assertIn("ImageGen", script)
        self.assertIn("modelscope download --model", script)
        self.assertIn("Qwen/Qwen-Image-2512", script)
        self.assertIn("Qwen/Qwen-Image-Edit-2511", script)
        self.assertIn("Qwen/Qwen-Image-Edit-2509", script)
        self.assertIn("nohup bash -lc", script)

    def test_src_ref_image_generation_script_uses_diffusers_model(self) -> None:
        script = Path("scripts/run_src_ref_image_generation_from_plan.sh").read_text(encoding="utf-8")
        helper = Path("scripts/generate_src_ref_images_from_plan.py").read_text(encoding="utf-8")

        self.assertIn("Qwen-Image-2512", script)
        self.assertIn("CUDA_VISIBLE_DEVICES", script)
        self.assertIn("generate_src_ref_images_from_plan.py", script)
        self.assertIn("DiffusionPipeline.from_pretrained", helper)
        self.assertIn('f"candidate_{candidate_index:03d}.png"', helper)
        self.assertIn("src_ref_image_generation_manifest", script)


if __name__ == "__main__":
    unittest.main()
