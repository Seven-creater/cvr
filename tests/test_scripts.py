from __future__ import annotations

import unittest
from pathlib import Path

from scripts.generate_grounded_sam2_video_masks import _mask_gate_errors


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
        self.assertIn("SRC_VIDEO_FOR_VACE", script)
        self.assertIn("maskedmerge", script)
        self.assertIn("review_inputs", script)
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
        self.assertIn("resolve_existing_path", script)
        self.assertIn("candidates = [path]", script)
        self.assertIn("FRAME_NUM=${FRAME_NUM:-81}", script)
        self.assertIn("VACE_CLIP_SECONDS=${VACE_CLIP_SECONDS:-5}", script)
        self.assertIn("VACE_SOURCE_FPS=${VACE_SOURCE_FPS:-16}", script)
        self.assertIn("VACE_DURATION_DRIFT_MAX=${VACE_DURATION_DRIFT_MAX:-0.5}", script)
        self.assertIn("reference_for_vace", script)
        self.assertIn("mask_for_vace", script)
        self.assertIn("tpad=stop_mode=clone", script)
        self.assertIn("trim=start_frame=0:end_frame=${FRAME_NUM}", script)
        self.assertIn("-count_frames", script)
        self.assertIn("frame_count", script)
        self.assertIn("expected_frame_num", script)
        self.assertIn("expected_fps", script)
        self.assertIn("preflight_report.json", script)
        self.assertIn("duration_metrics.json", script)
        self.assertIn("vace_command.json", script)
        self.assertIn("replacement_target_prompt_conflicts_with_source_state", script)
        self.assertIn("object_replacement_breaks_support_contact", script)
        self.assertIn("target_instance_description", script)
        self.assertIn("VIDEO_MASK_SEMANTICS_VERSION = 2", script)
        self.assertIn("VIDEO_MASK_POLARITY = \"white_generate_black_preserve\"", script)
        self.assertIn("mask_semantics_version", script)
        self.assertIn("mask_polarity", script)
        self.assertIn("mask manifest row has stale/missing mask_semantics_version", script)
        self.assertIn("mask manifest query does not match current plan", script)
        self.assertIn("background inverse mask appears to edit the subject", script)
        self.assertIn("low_contrast_dark_clothing_color_edit", script)
        self.assertIn("return \"torso clothing\"", script)
        self.assertIn("src_video_contact.jpg", script)
        self.assertIn("post_vace_verdict.json", script)
        self.assertIn("semantic_gate_family", script)
        self.assertIn("object_replacement", script)
        self.assertIn("raw_target_contact.jpg", script)
        self.assertIn("target_contact.jpg", script)
        self.assertIn("raw VACE target duration drift", script)
        self.assertNotIn("-shortest \"$TARGET_VIDEO\"", script)
        self.assertIn("src_mask", script)
        self.assertNotIn("Negative constraints:", script)
        self.assertIn("synthetic_visual_candidate_pairs.jsonl", script)
        self.assertIn("synthetic_visual_target_manifest.jsonl", script)

    def test_vace_visual_batch_script_runs_plan_ids_and_collects_pairs(self) -> None:
        script = Path("scripts/run_vace_visual_batch_from_plan.sh").read_text(encoding="utf-8")

        self.assertIn("PLAN_IDS", script)
        self.assertIn("TOP_K", script)
        self.assertIn("run_vace_visual_synthetic_smoke.sh", script)
        self.assertIn("bash scripts/run_vace_visual_synthetic_smoke.sh", script)
        self.assertIn("mapfile -t SELECTED_PLAN_IDS", script)
        self.assertIn("< /dev/null", script)
        self.assertIn("MASK_MANIFEST", script)
        self.assertIn("--mask-manifest", script)
        self.assertIn("SRC_REF_SELECTION", script)
        self.assertIn("--src-ref-selection", script)
        self.assertIn("FRAME_NUM=${FRAME_NUM:-81}", script)
        self.assertIn("VACE_CLIP_SECONDS=${VACE_CLIP_SECONDS:-5}", script)
        self.assertIn("VACE_SOURCE_FPS=${VACE_SOURCE_FPS:-16}", script)
        self.assertIn("--frame-num", script)
        self.assertIn("--vace-clip-seconds", script)
        self.assertIn("--vace-source-fps", script)
        self.assertIn("synthetic_visual_candidate_pairs.jsonl", script)
        self.assertIn("synthetic_visual_target_manifest.jsonl", script)
        self.assertIn("batch_generation_report.md", script)

    def test_vace_masked_batch_script_invokes_nested_script_with_bash(self) -> None:
        script = Path("scripts/run_vace_masked_visual_batch_from_plan.sh").read_text(encoding="utf-8")

        self.assertIn("bash scripts/run_vace_visual_batch_from_plan.sh", script)
        self.assertNotIn("\nscripts/run_vace_visual_batch_from_plan.sh", script)
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
        self.assertIn("min_detected_keyframe_box_coverage", helper)
        self.assertIn("mask_semantics_version", helper)
        self.assertIn("mask_polarity", helper)
        self.assertIn("sampled_frame_indices", helper)
        self.assertIn("detected_keyframe_index", helper)
        self.assertIn("reference_frame_count", helper)
        self.assertIn("generator_commit", helper)
        self.assertIn("visible_span_ratio", helper)
        self.assertIn("background_editable_ratio", helper)
        self.assertIn("subject_overlap_ratio", helper)
        self.assertIn("mask_target_instance_alignment", helper)
        self.assertIn("protected_overlap_queries", helper)
        self.assertIn("protected_overlap_ratio_max", helper)
        self.assertIn("min_protected_detections", helper)

    def test_grounded_sam2_mask_gate_rejects_low_replacement_coverage(self) -> None:
        errors = _mask_gate_errors(
            {"min_coverage_ratio": 0.01, "max_coverage_ratio": 0.15, "mask_not_empty_all_frames": True},
            {
                "mask_coverage_ratio_avg": 0.0025,
                "mask_temporal_stability": 0.95,
                "mask_nonempty_frame_ratio": 1.0,
            },
        )

        self.assertTrue(any("avg_coverage" in error and "< min" in error for error in errors))

    def test_grounded_sam2_mask_gate_rejects_weak_background_subject_mask(self) -> None:
        errors = _mask_gate_errors(
            {
                "min_coverage_ratio": 0.20,
                "max_coverage_ratio": 0.90,
                "min_detected_keyframe_box_coverage": 0.10,
                "max_subject_overlap_ratio": 0.20,
                "min_background_editable_ratio": 0.20,
                "mask_not_empty_all_frames": True,
            },
            {
                "mask_coverage_ratio_avg": 0.04,
                "detected_keyframe_box_coverage": 0.02,
                "subject_overlap_ratio": 0.35,
                "background_editable_ratio": 0.04,
                "mask_temporal_stability": 0.95,
                "mask_nonempty_frame_ratio": 1.0,
            },
        )

        self.assertTrue(any("avg_coverage" in error for error in errors))
        self.assertTrue(any("detected_keyframe_box_coverage" in error for error in errors))
        self.assertTrue(any("subject_overlap_ratio" in error for error in errors))
        self.assertTrue(any("background_editable_ratio" in error for error in errors))

    def test_grounded_sam2_mask_gate_rejects_protected_overlap_for_clothing(self) -> None:
        errors = _mask_gate_errors(
            {
                "min_coverage_ratio": 0.03,
                "max_coverage_ratio": 0.30,
                "max_protected_overlap_ratio": 0.18,
                "require_protected_overlap_metrics": True,
            },
            {
                "mask_coverage_ratio_avg": 0.12,
                "mask_temporal_stability": 0.95,
                "mask_nonempty_frame_ratio": 1.0,
                "protected_overlap_ratio_max": 0.31,
            },
        )

        self.assertTrue(any("protected_overlap_ratio" in error for error in errors))

    def test_grounded_sam2_mask_gate_rejects_missing_protected_detections_for_clothing(self) -> None:
        errors = _mask_gate_errors(
            {
                "min_coverage_ratio": 0.03,
                "max_coverage_ratio": 0.30,
                "min_protected_detections": 2,
            },
            {
                "mask_coverage_ratio_avg": 0.12,
                "mask_temporal_stability": 0.95,
                "mask_nonempty_frame_ratio": 1.0,
                "protected_overlap": [{"query": "face", "status": "detected", "overlap_ratio": 0.02}],
            },
        )

        self.assertTrue(any("protected_detection_count" in error for error in errors))

    def test_vace_smoke_script_lints_clothing_prompt_conflicts(self) -> None:
        script = Path("scripts/run_vace_visual_synthetic_smoke.sh").read_text(encoding="utf-8")

        self.assertIn("target_prompt_uses_operation_instruction_for_clothing_edit", script)
        self.assertIn("target_prompt_preserves_source_clothing", script)
        self.assertIn("structural_clothing_tryon_required", script)
        self.assertIn("target_prompt_contains_add_only_no", script)
        self.assertIn("replacement_target_prompt_uses_add_instead_of_replace", script)
        self.assertIn("replacement_target_prompt_missing_replace", script)
        self.assertIn("black_jacket_target_prompt_missing_open_black_long_sleeved_jacket", script)
        self.assertIn("black_jacket_target_prompt_forbidden_marker", script)
        self.assertIn("semantic_gate_required", script)
        self.assertIn("import re", script)

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
        helper = Path("scripts/generate_grounded_sam2_video_masks.py").read_text(encoding="utf-8")
        self.assertIn("_sample_keyframe_indices", helper)
        self.assertIn("mask gate failed", helper)
        self.assertIn("mask_nonempty_frame_ratio", helper)

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
        self.assertIn("ANNOTATE_CONCURRENCY=${ANNOTATE_CONCURRENCY:-1}", script)
        self.assertIn("--annotate-concurrency", script)
        self.assertIn("--concurrency \"$ANNOTATE_CONCURRENCY\"", script)
        self.assertIn("VACE_GPU_IDS=${VACE_GPU_IDS:-2,3,4,5}", script)
        self.assertIn("VACE_FRAME_NUM=${VACE_FRAME_NUM:-81}", script)
        self.assertIn("VACE_CLIP_SECONDS=${VACE_CLIP_SECONDS:-5}", script)
        self.assertIn("VACE_SOURCE_FPS=${VACE_SOURCE_FPS:-16}", script)
        self.assertIn("--vace-frame-num", script)
        self.assertIn("--vace-clip-seconds", script)
        self.assertIn("--vace-source-fps", script)
        self.assertIn("--frame-num \"$VACE_FRAME_NUM\"", script)
        self.assertIn("--vace-clip-seconds \"$VACE_CLIP_SECONDS\"", script)
        self.assertIn("--vace-source-fps \"$VACE_SOURCE_FPS\"", script)
        self.assertIn("plan-stable-omni-clips", script)
        self.assertIn("cache-reference-understandings", script)
        self.assertIn("plan-src-ref-images", script)
        self.assertIn("run_src_ref_image_generation_from_plan.sh", script)
        self.assertIn("bash scripts/run_src_ref_image_generation_from_plan.sh", script)
        self.assertIn("--image-gpu-ids", script)
        self.assertIn("IMAGE_GEN_GPU_IDS", script)
        self.assertIn("--image-device-map", script)
        self.assertIn("IMAGE_GEN_DEVICE_MAP", script)
        self.assertIn("select-src-ref-images", script)
        self.assertIn("PLANNING_MODE=${PLANNING_MODE:-production}", script)
        self.assertIn("plan-video-edits", script)
        self.assertIn("--planning-mode \"$PLANNING_MODE\"", script)
        self.assertIn("--planner-cache-path \"$VIDEO_EDIT_PLANNER_CACHE\"", script)
        self.assertIn("plan-video-masks", script)
        self.assertIn("run_grounded_sam2_video_masks.sh", script)
        self.assertIn("bash scripts/run_grounded_sam2_video_masks.sh", script)
        self.assertIn("run_vace_masked_visual_batch_from_plan.sh", script)
        self.assertIn("bash scripts/run_vace_masked_visual_batch_from_plan.sh", script)
        self.assertIn("detective-annotate-clips", script)
        self.assertIn("validate-known-pairs", script)
        self.assertIn("build_synthetic_manual_review_bundle.sh", script)
        self.assertIn("bash scripts/build_synthetic_manual_review_bundle.sh", script)
        self.assertNotRegex(script, r"(?m)^\s{2}scripts/[^ ]+\.sh")
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
        self.assertIn("[src-ref-gen] CUDA_VISIBLE_DEVICES", script)
        self.assertIn("--device-map", script)
        self.assertIn("--low-cpu-mem-usage", script)
        self.assertIn("--background-width", script)
        self.assertIn("--background-height", script)
        self.assertIn("generate_src_ref_images_from_plan.py", script)
        self.assertIn("DiffusionPipeline.from_pretrained", helper)
        self.assertIn('"device_map"', helper)
        self.assertIn('"width"', helper)
        self.assertIn('"height"', helper)
        self.assertIn("true_cfg_scale", helper)
        self.assertIn('f"candidate_{candidate_index:03d}.png"', helper)
        self.assertIn("src_ref_image_generation_manifest", script)


if __name__ == "__main__":
    unittest.main()
