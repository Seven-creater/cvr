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
        self.assertIn("synthetic_visual_candidate_pairs.jsonl", script)
        self.assertIn("synthetic_visual_target_manifest.jsonl", script)

    def test_vace_visual_batch_script_runs_plan_ids_and_collects_pairs(self) -> None:
        script = Path("scripts/run_vace_visual_batch_from_plan.sh").read_text(encoding="utf-8")

        self.assertIn("PLAN_IDS", script)
        self.assertIn("TOP_K", script)
        self.assertIn("run_vace_visual_synthetic_smoke.sh", script)
        self.assertIn("synthetic_visual_candidate_pairs.jsonl", script)
        self.assertIn("synthetic_visual_target_manifest.jsonl", script)
        self.assertIn("batch_generation_report.md", script)
        self.assertIn("This script does not start Omni", script)

    def test_manual_review_bundle_script_calls_review_bundle_command(self) -> None:
        script = Path("scripts/build_synthetic_manual_review_bundle.sh").read_text(encoding="utf-8")

        self.assertIn("build-review-bundle", script)
        self.assertIn("--pairs-path", script)
        self.assertIn("--output-dir", script)
        self.assertIn("reference.mp4, target.mp4, review.md, and metadata.json", script)


if __name__ == "__main__":
    unittest.main()
