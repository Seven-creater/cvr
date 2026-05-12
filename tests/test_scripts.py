from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from scripts.generate_grounded_sam2_video_masks import (
    _find_grounding_dino_checkpoint,
    _mask_gate_errors,
    _mask_quality_tier,
    _sample_detection_frame_indices,
    _select_anchor_detections,
    _visible_spans_from_masks,
)


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

    def test_omni_detective_script_accepts_natural_pair_controls(self) -> None:
        script = Path("scripts/run_omni_detective_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("CONCURRENCY=${CONCURRENCY:-1}", script)
        self.assertIn("MAX_ACCEPTED_PAIRS=${MAX_ACCEPTED_PAIRS:-10}", script)
        self.assertIn("MAX_PROPOSALS=${MAX_PROPOSALS:-40}", script)
        self.assertIn("MAX_MINED_CANDIDATES=${MAX_MINED_CANDIDATES:-240}", script)
        self.assertIn("MODEL=${MODEL:-qwen3-omni}", script)
        self.assertIn("ANNOTATION_MAX_PASSES=${ANNOTATION_MAX_PASSES:-3}", script)
        self.assertIn("ANNOTATION_PASS_TIMEOUT_SECONDS=${ANNOTATION_PASS_TIMEOUT_SECONDS:-900}", script)
        self.assertIn("MINE_CANDIDATES_TIMEOUT_SECONDS=${MINE_CANDIDATES_TIMEOUT_SECONDS:-120}", script)
        self.assertIn("PROPOSE_TIMEOUT_SECONDS=${PROPOSE_TIMEOUT_SECONDS:-900}", script)
        self.assertIn("PAIR_REQUEST_TIMEOUT_SECONDS=${PAIR_REQUEST_TIMEOUT_SECONDS:-90}", script)
        self.assertIn("ZERO_ACCEPTED_STOP_AFTER=${ZERO_ACCEPTED_STOP_AFTER:-10}", script)
        self.assertIn("START_STAGE=${START_STAGE:-plan}", script)
        self.assertIn("ACCEPTANCE_PROFILE=${ACCEPTANCE_PROFILE:-final}", script)
        self.assertIn("ALLOW_PARTIAL_ANNOTATIONS=${ALLOW_PARTIAL_ANNOTATIONS:-0}", script)
        self.assertIn("--concurrency", script)
        self.assertIn("--max-accepted-pairs", script)
        self.assertIn("--max-proposals", script)
        self.assertIn("--max-mined-candidates", script)
        self.assertIn("--annotation-max-passes", script)
        self.assertIn("--annotation-pass-timeout-seconds", script)
        self.assertIn("--mine-candidates-timeout-seconds", script)
        self.assertIn("--propose-timeout-seconds", script)
        self.assertIn("--pair-request-timeout-seconds", script)
        self.assertIn("--zero-accepted-stop-after", script)
        self.assertIn("--acceptance-profile", script)
        self.assertIn("--start-stage", script)
        self.assertIn("--allow-partial-annotations", script)
        self.assertIn('--concurrency "$CONCURRENCY"', script)
        self.assertIn('--max-accepted-pairs "$MAX_ACCEPTED_PAIRS"', script)
        self.assertIn('--max-proposals "$MAX_PROPOSALS"', script)
        self.assertIn('--timeout-seconds "$PAIR_REQUEST_TIMEOUT_SECONDS"', script)
        self.assertIn('--zero-accepted-stop-after "$ZERO_ACCEPTED_STOP_AFTER"', script)
        self.assertIn('--acceptance-profile "$ACCEPTANCE_PROFILE"', script)
        self.assertIn("annotation pass $ANNOTATION_PASS/$ANNOTATION_MAX_PASSES", script)
        self.assertIn("unique_done=$ANNOTATION_DONE_COUNT/$ANNOTATION_TARGET_COUNT", script)
        self.assertIn("jsonl_unique_clip_count", script)
        self.assertIn("annotation coverage unique_done=", script)
        self.assertIn("annotation incomplete by unique clip_id count", script)
        self.assertIn("mine-pair-candidates", script)
        self.assertIn("--mined-candidates-path", script)
        self.assertIn("candidate_mining_report.md", script)
        self.assertIn("probe_omni_model", script)
        self.assertIn("served_models=", script)
        self.assertIn("is not served by", script)
        self.assertIn("## Candidate Funnel", script)
        self.assertIn("mine-candidates exit=", script)
        self.assertIn("run_with_timeout", script)
        self.assertIn("timed out after", script)
        self.assertIn("propose exit=$PROPOSE_STATUS", script)
        self.assertIn("judged and 0 accepted", script)
        self.assertIn('stage_enabled "annotate"', script)
        self.assertIn('stage_enabled "propose"', script)
        self.assertIn('require_file "$RUN_ROOT/detective_annotations.jsonl"', script)
        self.assertIn("build-review-bundle", script)
        self.assertIn("build-diagnostic-bundle", script)
        self.assertIn("manual_review_bundle", script)
        self.assertIn("diagnostic_bundle", script)
        self.assertIn("exploration_review_summary.md", script)

    def test_single_source_omni_pair_script_runs_one_source_pipeline(self) -> None:
        script = Path("scripts/run_single_source_omni_pair_pilot.sh").read_text(encoding="utf-8")

        self.assertIn("select-single-source-video", script)
        self.assertIn("plan-single-source-clips", script)
        self.assertIn("extract-clips", script)
        self.assertIn("detective-annotate-clips", script)
        self.assertIn("mine-single-source-pairs", script)
        self.assertIn("propose-single-source-pairs", script)
        self.assertIn("--pair-candidates-path", script)
        self.assertIn("--whole-annotation-path", script)
        self.assertIn("build-single-source-review-bundle", script)
        self.assertIn("MODEL=${MODEL:-qwen3-omni}", script)
        self.assertIn("DATASET=${DATASET:-daily_omni}", script)
        self.assertIn("SEGMENT_SECONDS=${SEGMENT_SECONDS:-6}", script)
        self.assertIn("ZERO_ACCEPTED_STOP_AFTER=${ZERO_ACCEPTED_STOP_AFTER:-10}", script)
        self.assertIn("MAX_PROPOSALS=${MAX_PROPOSALS:-15}", script)
        self.assertIn("ANNOTATION_TIMEOUT_SECONDS=${ANNOTATION_TIMEOUT_SECONDS:-900}", script)
        self.assertIn("SOURCE_SELECTION_MODE=${SOURCE_SELECTION_MODE:-random}", script)
        self.assertIn("SOURCE_SELECTION_RANDOM_SEED=${SOURCE_SELECTION_RANDOM_SEED:-}", script)
        self.assertIn("SOURCE_SELECTION_TOP_K=${SOURCE_SELECTION_TOP_K:-1}", script)
        self.assertIn("SOURCE_SELECTION_SCAN_LIMIT=${SOURCE_SELECTION_SCAN_LIMIT:-500}", script)
        self.assertIn("SOURCE_SELECTION_MAX_ELIGIBLE=${SOURCE_SELECTION_MAX_ELIGIBLE:-24}", script)
        self.assertIn("OMNI_SOURCE_SELECTION=${OMNI_SOURCE_SELECTION:-0}", script)
        self.assertIn("--source-selection-mode", script)
        self.assertIn("--dataset \"$DATASET\"", script)
        self.assertIn("extracted_single_source_whole.jsonl", script)
        self.assertIn("--source-selection-random-seed", script)
        self.assertIn("--source-selection-top-k", script)
        self.assertIn("--source-selection-scan-limit", script)
        self.assertIn("--source-selection-max-eligible", script)
        self.assertIn("--omni-source-selection", script)
        self.assertIn("fast local source selection mode=", script)
        self.assertIn("single_source_review_bundle", script)

    def test_single_source_omni_batch_script_runs_dual_dataset_sources(self) -> None:
        script = Path("scripts/run_single_source_omni_batch.sh").read_text(encoding="utf-8")

        self.assertIn("run_single_source_omni_pair_pilot.sh", script)
        self.assertIn("WORLDSENSE_ROOT=${WORLDSENSE_ROOT:-$ROOT/raw_datasets/worldsense/_extracted}", script)
        self.assertIn("SEGMENT_SECONDS=${SEGMENT_SECONDS:-6}", script)
        self.assertIn("DAILY_SOURCE_COUNT=${DAILY_SOURCE_COUNT:-5}", script)
        self.assertIn("WORLDSENSE_SOURCE_COUNT=${WORLDSENSE_SOURCE_COUNT:-5}", script)
        self.assertIn("MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-2}", script)
        self.assertIn("videos_chunk_*/videos/*.mp4", script)
        self.assertIn("source_window_start_seconds", script)
        self.assertIn("source_window_duration_seconds", script)
        self.assertIn("batch_source_manifest.jsonl", script)
        self.assertIn("batch_accepted_pairs.jsonl", script)
        self.assertIn("batch_summary.md", script)
        self.assertIn("manual_review/accepted", script)
        self.assertIn("manual_review/diagnostic", script)
        self.assertIn("tee \"$job_root/logs/job.log\"", script)
        self.assertIn("final_omni_quality_score", Path("app/composed_data.py").read_text(encoding="utf-8"))
        self.assertIn("served_models=", script)
        self.assertIn("is not served by", script)

    def test_audio_lines_single_source_reuse_script_is_safe_and_parallel(self) -> None:
        script = Path("scripts/run_audio_lines_single_source_reuse.sh").read_text(encoding="utf-8")

        self.assertIn("prepare_existing_args=(", script)
        self.assertIn("prepare-existing", script)
        self.assertIn("clips/single_source", script)
        self.assertIn("detective-annotate-clips", script)
        self.assertIn("mine-single-source-pairs", script)
        self.assertIn("propose-single-source-pairs", script)
        self.assertIn("--audio-dataset-line", script)
        self.assertIn("visual_audio_anchor", script)
        self.assertIn("speech_audio_content", script)
        self.assertIn("MODEL=${MODEL:-qwen3-omni-30b-a3b-instruct}", script)
        self.assertIn("resolved_model_alias=", script)
        self.assertIn("SHARD_TIMEOUT_SECONDS=${SHARD_TIMEOUT_SECONDS:-3600}", script)
        self.assertIn("timeout \"$SHARD_TIMEOUT_SECONDS\" python3 -m app.composed_data propose-single-source-pairs", script)
        self.assertIn("failed_or_timed_out_shards", script)
        self.assertIn("AUDIO_LINE_QUALITY_PROFILE=${AUDIO_LINE_QUALITY_PROFILE:-default}", script)
        self.assertIn("--audio-line-quality-profile \"$AUDIO_LINE_QUALITY_PROFILE\"", script)
        self.assertIn("B_ACCEPTANCE_PROFILE=${B_ACCEPTANCE_PROFILE:-exploration}", script)
        self.assertIn("--acceptance-profile", script)
        self.assertIn("b_audio_context_cvr", script)
        self.assertIn("--reuse-run-root", script)
        self.assertIn("--skip-annotation-refresh", script)
        self.assertIn("--run-b-only", script)
        self.assertIn("--force-audio-focused-refresh", script)
        self.assertIn("--fresh-annotations", script)
        self.assertIn("--no-annotation-reuse", script)
        self.assertIn("MAX_CLIPS=${MAX_CLIPS:-0}", script)
        self.assertIn("--max-clips", script)
        self.assertIn("ANNOTATION_SEARCH_ROOTS=${ANNOTATION_SEARCH_ROOTS:-}", script)
        self.assertIn("--annotation-search-root PATH", script)
        self.assertIn('if [ "${#annotation_search_args[@]}" -gt 0 ]; then', script)
        self.assertIn('prepare_existing_args+=("${annotation_search_args[@]}")', script)
        self.assertLess(
            script.index('if [ "${#annotation_search_args[@]}" -gt 0 ]; then'),
            script.index('elif [ "$FRESH_ANNOTATIONS" = "1" ]; then'),
        )
        self.assertNotIn("$(if [ \"$FRESH_ANNOTATIONS\" = \"1\" ]; then printf '%s' '--no-annotation-reuse'; fi)", script)
        self.assertIn("A_CANDIDATE_MODE=${A_CANDIDATE_MODE:-hybrid}", script)
        self.assertIn("--a-candidate-mode \"$A_CANDIDATE_MODE\"", script)
        self.assertIn("B_CANDIDATE_MODE=${B_CANDIDATE_MODE:-hybrid}", script)
        self.assertIn("--b-candidate-mode \"$B_CANDIDATE_MODE\"", script)
        self.assertIn("v5_audio_primary", script)
        self.assertIn("--audio-focused", script)
        self.assertIn("OMNI_TRANSIENT_RETRIES=${OMNI_TRANSIENT_RETRIES:-2}", script)
        self.assertIn("--omni-retries \"$OMNI_TRANSIENT_RETRIES\"", script)
        self.assertIn("--fail-on-transient-omni-errors", script)
        self.assertIn("PROPOSE_SHARDS=${PROPOSE_SHARDS:-16}", script)
        self.assertIn("PROPOSE_PARALLEL_JOBS=${PROPOSE_PARALLEL_JOBS:-8}", script)
        self.assertIn("CONCURRENCY=${CONCURRENCY:-4}", script)
        self.assertIn("accepted_progress_", script)
        self.assertIn("rejected_progress_", script)
        self.assertIn("manual_review/A", script)
        self.assertIn("manual_review/B", script)
        self.assertNotIn("VACE", script)
        self.assertNotIn("modelscope download", script)
        self.assertNotIn("vllm", script)
        self.assertNotIn("8092", script)

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

    def test_composed_avigate_script_supports_sample_sized_runs(self) -> None:
        script = Path("scripts/run_composed_avigate_smoke20.sh").read_text(encoding="utf-8")
        wrapper = Path("scripts/run_composed_avigate_400.sh").read_text(encoding="utf-8")

        self.assertIn("RUN_ROOT=${RUN_ROOT:-}", script)
        self.assertIn("composed_avigate_eval${SAMPLE_SIZE}_", script)
        self.assertIn('STAGED_ROOT="$RUN_ROOT/staged"', script)
        self.assertIn('--sample-size "$SAMPLE_SIZE"', script)
        self.assertIn("SAMPLE_SIZE=${SAMPLE_SIZE:-400}", wrapper)
        self.assertIn("run_composed_avigate_smoke20.sh", wrapper)

    def test_e5_cvr_eval_script_is_e5_only_and_uses_existing_model(self) -> None:
        script = Path("scripts/run_e5_cvr_eval.sh").read_text(encoding="utf-8")

        self.assertIn("app.e5_cvr_eval", script)
        self.assertIn("e5-omni-7B", script)
        self.assertIn("require_path \"e5 config\" \"$E5_MODEL/config.json\"", script)
        self.assertIn("--triplets-jsonl \"$TRIPLETS_JSONL\"", script)
        self.assertIn("QUERY_MODE=${QUERY_MODE:-composed}", script)
        self.assertIn("--query-mode \"$QUERY_MODE\"", script)
        self.assertIn("REFERENCE_AUDIO_MODE=${REFERENCE_AUDIO_MODE:-original}", script)
        self.assertIn("--reference-audio-mode \"$REFERENCE_AUDIO_MODE\"", script)
        self.assertIn("VIDEO_AUDIO_MODE=${VIDEO_AUDIO_MODE:-on}", script)
        self.assertIn("--video-audio-mode \"$VIDEO_AUDIO_MODE\"", script)
        self.assertIn("load_audio_from_video=$LOAD_AUDIO_FROM_VIDEO", script)
        self.assertIn("use_audio_in_video=false", script)
        self.assertIn("processor_video_kwargs_sanitizer=runtime", script)
        self.assertIn("e5_audio_on", script)
        self.assertIn("e5_audio_off", script)
        self.assertIn("video_only_ref_${REFERENCE_AUDIO_MODE}_eval_", script)
        self.assertIn("command -v \"$FFMPEG\"", script)
        self.assertIn("--target-index-dir \"$TARGET_INDEX_DIR\"", script)
        self.assertIn("video_only_eval_", script)
        self.assertIn("VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}", script)
        self.assertNotIn("modelscope download", script)
        self.assertNotIn("vllm.entrypoints.openai.api_server", script)
        self.assertNotIn("8092", script)

    def test_e5_three_data_mixed_script_runs_three_modes_and_groups_results(self) -> None:
        script = Path("scripts/run_e5_three_data_mixed_eval.sh").read_text(encoding="utf-8")

        self.assertIn("merged_all/triplets.jsonl", script)
        self.assertIn("EXPECTED_COUNT=${EXPECTED_COUNT:-1697}", script)
        self.assertIn("PARALLEL_MODES=${PARALLEL_MODES:-0}", script)
        self.assertIn("--parallel-modes", script)
        self.assertIn("--gpu-ids ID1,ID2,ID3", script)
        self.assertIn("vta_audio_on", script)
        self.assertIn('--query-mode "$query_mode"', script)
        self.assertIn('--video-audio-mode "$video_audio_mode"', script)
        self.assertIn("vt_audio_off", script)
        self.assertIn("va_video_only_audio_on", script)
        self.assertIn('"$RUN_ROOT/vta_audio_on" "$GPU_ID" composed on', script)
        self.assertIn('"$RUN_ROOT/vt_audio_off" "$GPU_ID" composed off', script)
        self.assertIn('"$RUN_ROOT/va_video_only_audio_on" \\', script)
        self.assertIn('"$GPU_ID" \\', script)
        self.assertIn('"$RUN_ROOT/vta_audio_on" "${MODE_GPUS[0]}" composed on', script)
        self.assertIn('"$RUN_ROOT/vt_audio_off" "${MODE_GPUS[1]}" composed off', script)
        self.assertIn('"$RUN_ROOT/va_video_only_audio_on" "${MODE_GPUS[2]}" video-only on', script)
        self.assertIn('--target-index-dir "$RUN_ROOT/vta_audio_on/target_index"', script)
        self.assertIn("mode_logs", script)
        self.assertIn("app.e5_three_data_eval", script)
        self.assertIn("comparison_by_dataset.md", script)
        self.assertNotIn("modelscope download", script)
        self.assertNotIn("vllm.entrypoints.openai.api_server", script)
        self.assertNotIn("8092", script)
        self.assertNotIn("8093", script)

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
        self.assertIn('"model": "ffmpeg-deterministic-composite" if not requires_vace else (wan_ckpt.name or str(wan_ckpt))', script)
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
        self.assertIn("VIDEO_MASK_SEMANTICS_VERSION = 3", script)
        self.assertIn("VIDEO_MASK_POLARITY = \"white_generate_black_preserve\"", script)
        self.assertIn("mask_semantics_version", script)
        self.assertIn("mask_polarity", script)
        self.assertIn("mask manifest row has stale/missing mask_semantics_version", script)
        self.assertIn("mask manifest query does not match current plan", script)
        self.assertIn("mask_generation_strategy", script)
        self.assertIn("adaptive_repair_v1", script)
        self.assertIn("diagnostic-only and not usable for VACE", script)
        self.assertIn("mask_quality_tier", script)
        self.assertIn("background inverse mask appears to edit the subject", script)
        self.assertIn("low_contrast_dark_clothing_color_edit", script)
        self.assertIn("multi_subject_background_mask_route_unsupported", script)
        self.assertIn("return \"torso clothing\"", script)
        self.assertIn("src_video_contact.jpg", script)
        self.assertIn("post_vace_verdict.json", script)
        self.assertIn("semantic_gate_family", script)
        self.assertIn("original room, windows, doors, walls, or brick wall must not remain visible", script)
        self.assertIn("blue tint, blue overlay, or style wash", script)
        self.assertIn("target_prompt_contains_source_background", script)
        self.assertIn("background_replace_contains_source_layout_or_lighting_lock", script)
        self.assertIn("background_replace_plain_masked_vace_disabled", script)
        self.assertIn("ALLOW_PLAIN_BACKGROUND_REPLACE", script)
        self.assertIn("VACE_BG_REPLACE_COMPOSITE_ROUTE", script)
        self.assertIn("BACKGROUND_REPLACE_ROUTE", script)
        self.assertIn("vace_bg_replace_composite_first_frame_mv2v", script)
        self.assertIn("deterministic_foreground_background_composite", script)
        self.assertIn("guided_composite_refine_vace", script)
        self.assertIn("composite-first-frame requires src_mask", script)
        self.assertIn("composite-first-frame requires at least one src_ref_image", script)
        self.assertIn("deterministic foreground/background composite requires src_mask", script)
        self.assertIn("build_deterministic_masked_composite.py", script)
        self.assertIn("post_composite_pre_omni_validation", script)
        self.assertIn("post_vace_or_composite_verdict.json", script)
        self.assertIn("composite_frame0.png", script)
        self.assertIn("composite_src_video_contact.jpg", script)
        self.assertIn("composite_src_mask_contact.jpg", script)
        self.assertIn("alphamerge", script)
        self.assertIn("black_frame0", script)
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
        self.assertIn("adaptive_repair_v1", helper)
        self.assertIn("sparse_full_length", helper)
        self.assertIn("visible_spans", helper)
        self.assertIn("detector_cascade", helper)
        self.assertIn("detection_attempts", helper)
        self.assertIn("anchor_frame_indices", helper)
        self.assertIn("prompt_type", helper)
        self.assertIn("mask_quality_tier", helper)
        self.assertIn("usable_for_vace", helper)
        self.assertIn("usable_for_vace_default", helper)
        self.assertIn("tier=`{row.get('mask_quality_tier')}`", helper)
        self.assertIn("visible_spans=`{len(row.get('visible_spans') or [])}`", helper)
        self.assertIn("_write_all_black_mask_video", helper)
        self.assertIn("_sample_detection_frame_indices", helper)

    def test_grounding_dino_checkpoint_rejects_huggingface_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "model.safetensors").write_bytes(b"not a torch checkpoint")
            (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "pytorch_model.bin").write_bytes(b"huggingface state dict placeholder")

            with self.assertRaisesRegex(FileNotFoundError, "HuggingFace-format"):
                _find_grounding_dino_checkpoint(checkpoint_dir)

    def test_grounding_dino_checkpoint_accepts_non_hf_torch_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "model.safetensors").write_bytes(b"not a torch checkpoint")
            expected = checkpoint_dir / "groundingdino_model.bin"
            expected.write_bytes(b"torch checkpoint placeholder")

            self.assertEqual(expected, _find_grounding_dino_checkpoint(checkpoint_dir))

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
                "min_foreground_subject_coverage_ratio": 0.04,
                "max_foreground_subject_coverage_ratio": 0.70,
                "min_foreground_subject_temporal_stability": 0.75,
                "min_foreground_subject_nonempty_frame_ratio": 0.90,
                "mask_not_empty_all_frames": True,
            },
            {
                "mask_coverage_ratio_avg": 0.04,
                "detected_keyframe_box_coverage": 0.02,
                "subject_overlap_ratio": 0.35,
                "background_editable_ratio": 0.04,
                "foreground_subject_coverage_ratio_avg": 0.96,
                "foreground_subject_temporal_stability": 0.40,
                "foreground_subject_nonempty_frame_ratio": 0.50,
                "mask_temporal_stability": 0.95,
                "mask_nonempty_frame_ratio": 1.0,
            },
        )

        self.assertTrue(any("avg_coverage" in error for error in errors))
        self.assertTrue(any("detected_keyframe_box_coverage" in error for error in errors))
        self.assertTrue(any("subject_overlap_ratio" in error for error in errors))
        self.assertTrue(any("background_editable_ratio" in error for error in errors))
        self.assertTrue(any("foreground_subject_coverage_ratio" in error for error in errors))
        self.assertTrue(any("foreground_subject_temporal_stability" in error for error in errors))
        self.assertTrue(any("foreground_subject_nonempty_frame_ratio" in error for error in errors))

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

    def test_adaptive_mask_sampling_uses_dense_frame_candidates(self) -> None:
        sampled = _sample_detection_frame_indices(101, max_samples=13)

        self.assertIn(0, sampled)
        self.assertIn(25, sampled)
        self.assertIn(50, sampled)
        self.assertIn(75, sampled)
        self.assertIn(100, sampled)
        self.assertGreater(len(sampled), 5)

    def test_adaptive_mask_selects_multiple_anchor_frames(self) -> None:
        anchors = _select_anchor_detections(
            [
                {"frame_idx": 10, "score": 0.2},
                {"frame_idx": 30, "score": 0.9},
                {"frame_idx": 70, "score": 0.7},
                {"frame_idx": 90, "score": 0.6},
            ],
            max_anchors=3,
        )

        self.assertEqual([30, 70, 90], [item["frame_idx"] for item in anchors])

    def test_adaptive_sparse_mask_records_visible_spans(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            self.skipTest(f"numpy unavailable: {exc}")

        masks = {
            2: np.ones((2, 2), dtype="uint8"),
            3: np.ones((2, 2), dtype="uint8"),
            7: np.ones((2, 2), dtype="uint8"),
        }

        self.assertEqual(
            [
                {"start_frame": 2, "end_frame": 3, "frame_count": 2, "coverage_avg": 1.0},
                {"start_frame": 7, "end_frame": 7, "frame_count": 1, "coverage_avg": 1.0},
            ],
            _visible_spans_from_masks(masks, 10),
        )

    def test_adaptive_mask_quality_tier_keeps_diagnostics_out_of_vace(self) -> None:
        self.assertEqual(
            "diagnostic_only",
            _mask_quality_tier(
                ["temporal_stability 0.2500 < min 0.7500"],
                {
                    "mask_coverage_ratio_avg": 0.2,
                    "mask_nonempty_frame_ratio": 0.25,
                    "mask_temporal_stability": 0.25,
                },
            ),
        )
        self.assertEqual(
            "excellent",
            _mask_quality_tier(
                [],
                {
                    "mask_coverage_ratio_avg": 0.5,
                    "mask_nonempty_frame_ratio": 1.0,
                    "mask_temporal_stability": 0.95,
                },
            ),
        )

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
        self.assertIn("failed_gate", helper)
        self.assertIn("mask_nonempty_frame_ratio", helper)

    def test_manual_review_bundle_script_calls_review_bundle_command(self) -> None:
        script = Path("scripts/build_synthetic_manual_review_bundle.sh").read_text(encoding="utf-8")

        self.assertIn("build-review-bundle", script)
        self.assertIn("--pairs-path", script)
        self.assertIn("--output-dir", script)
        self.assertIn("reference.mp4, target.mp4, review.md, metadata.json", script)

    def test_audio_matters_natural_script_uses_natural_omni_pipeline(self) -> None:
        script = Path("scripts/run_audio_matters_natural_omni.sh").read_text(encoding="utf-8")

        self.assertIn("plan-detective-clips", script)
        self.assertIn("extract-clips", script)
        self.assertIn("detective-annotate-clips", script)
        self.assertIn("--reuse-run-root", script)
        self.assertIn("--audio-workers", script)
        self.assertIn("app.audio_matters_natural mine-candidates", script)
        self.assertIn("split-candidates", script)
        self.assertIn("propose-group-pairs", script)
        self.assertIn("merge-proposals", script)
        self.assertIn("--propose-shards", script)
        self.assertIn("--propose-parallel-jobs", script)
        self.assertIn("ACCEPTANCE_PROFILE=${ACCEPTANCE_PROFILE:-audio_matters}", script)
        self.assertIn("--accepted-progress-path", script)
        self.assertIn("accepted_audio_matters_pairs.progress.jsonl", script)
        self.assertIn("STRICT_VISUAL_ANCHOR=${STRICT_VISUAL_ANCHOR:-1}", script)
        self.assertIn("--no-strict-audio-matters-visual-anchor", script)
        self.assertIn("--rejected-progress-path", script)
        self.assertIn("audio_matters_rejected_with_reasons.jsonl", script)
        self.assertIn("ACCEPTED_SAMPLE", Path("app/composed_data.py").read_text(encoding="utf-8"))
        self.assertIn("REJECTED_PROPOSAL", Path("app/composed_data.py").read_text(encoding="utf-8"))
        self.assertIn("GENERATED_TRIPLET", Path("app/audio_matters_natural.py").read_text(encoding="utf-8"))
        self.assertIn('tee "$SHARD_LOG"', script)
        self.assertIn("--skip-review-bundle", script)
        self.assertIn("audio_matters_triplets.jsonl", script)
        self.assertIn("http://127.0.0.1:8093/v1", script)
        self.assertNotIn("/Demo/test/data", script)
        self.assertNotIn("build_composed_triplets.sh", script)
        self.assertNotIn("modelscope download", script)
        self.assertNotIn("8092", script)
        self.assertNotIn("vace", script.lower())

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
