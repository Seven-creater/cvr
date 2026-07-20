# Audio-CVR 自动复核与 150 条 Benchmark 冻结说明

日期：2026-07-20

## 1. 目标与口径

本流程合并旧 640 条与 fresh 485 条 accepted records，经稳定去重、独立 Omni 三阶段复核、20% 重复复核和 source-disjoint 重切分，尝试冻结：

```text
test_main = 150
sound_event = 90
music = 30
contextual speech = 30
```

如果高质量候选不足，流程会失败并报告实际数量，不会降低 audio-only、video-only、full-AV、ASR 或泄漏门槛来凑数。

论文中应称为：

> automatically curated and model-verified benchmark

不能称为 human-validated benchmark。

## 2. 三个 CLI 入口

### 2.1 合并、过滤、去重和准备复核池

```bash
python3 -m app.audio_cvr_paper_experiment prepare-automatic-benchmark-review \
  --input-path <OLD_RUN>/b_all_audio_cvr_triplets.jsonl \
  --input-path <FRESH_RUN>/b_all_audio_cvr_triplets.jsonl \
  --output-dir <WORK_DIR>/review_pool \
  --review-pool-targets sound_event=180,music=70,speech_topic_in_video_context=180 \
  --max-per-source 2 \
  --random-seed 20260720
```

主要产物：

```text
combined_accepted_pool.jsonl
combined_pool_deduplicated.jsonl
automatic_review_candidates.jsonl
combined_pool_summary.json
```

`legacy asr_degeneracy_risk=0.45` 只保存为历史字段，不参与新 benchmark 判断。选择器不读取 E5 score、rank、checkpoint 或既有测试结果。

### 2.2 分片执行两轮 Omni 复核

第一轮对所有候选执行 audio-only、muted-video、full-AV 和 contextual/ASR audit：

```bash
python3 -m app.audio_cvr_paper_experiment review-benchmark-omni \
  --candidate-path <WORK_DIR>/review_pool/automatic_review_candidates.jsonl \
  --output-path <WORK_DIR>/reviews/pass1/shard_<SHARD_INDEX>.jsonl \
  --media-root <REPO_ROOT> \
  --cache-dir <WORK_DIR>/media_cache \
  --base-url http://127.0.0.1:8093/v1 \
  --api-key EMPTY \
  --model <OMNI_MODEL> \
  --review-pass-id 1 \
  --shard-index <SHARD_INDEX> \
  --shard-count <SHARD_COUNT> \
  --timeout-seconds 180 \
  --omni-retries 2 \
  --resume
```

第二轮只复核第一轮 pass 中由固定 seed 选出的 20%。第二轮保持 reference/target 语义标签不变，但改变媒体展示顺序与检查顺序，用于测量审核稳定性：

```bash
PASS1_ARGS=()
for path in <WORK_DIR>/reviews/pass1/shard_*.jsonl; do
  PASS1_ARGS+=(--pass1-review-path "$path")
done

python3 -m app.audio_cvr_paper_experiment review-benchmark-omni \
  --candidate-path <WORK_DIR>/review_pool/automatic_review_candidates.jsonl \
  --output-path <WORK_DIR>/reviews/pass2/shard_<SHARD_INDEX>.jsonl \
  --media-root <REPO_ROOT> \
  --cache-dir <WORK_DIR>/media_cache \
  --base-url http://127.0.0.1:8093/v1 \
  --api-key EMPTY \
  --model <OMNI_MODEL> \
  --review-pass-id 2 \
  "${PASS1_ARGS[@]}" \
  --repeat-review-fraction 0.20 \
  --random-seed 20260720 \
  --shard-index <SHARD_INDEX> \
  --shard-count <SHARD_COUNT> \
  --timeout-seconds 180 \
  --omni-retries 2 \
  --resume
```

`--resume` 会保留成功记录，并重新尝试上次标记为 `error` 的记录。任一轮 reject、uncertain、关键字段不一致或 speech role 不一致，都不能进入 test-main。

### 2.3 冻结测试集并重切 train/val/test

```bash
PASS1_ARGS=()
PASS2_ARGS=()
for path in <WORK_DIR>/reviews/pass1/shard_*.jsonl; do
  PASS1_ARGS+=(--pass1-review-path "$path")
done
for path in <WORK_DIR>/reviews/pass2/shard_*.jsonl; do
  PASS2_ARGS+=(--pass2-review-path "$path")
done

python3 -m app.audio_cvr_paper_experiment finalize-automatic-benchmark \
  --combined-pool-path <WORK_DIR>/review_pool/combined_pool_deduplicated.jsonl \
  --candidate-path <WORK_DIR>/review_pool/automatic_review_candidates.jsonl \
  "${PASS1_ARGS[@]}" \
  "${PASS2_ARGS[@]}" \
  --output-dir <WORK_DIR>/benchmark_v1 \
  --subtype-targets sound_event=90,music=30,speech_topic_in_video_context=30 \
  --validation-targets sound_event=45,music=15,speech_topic_in_video_context=15 \
  --repeat-review-fraction 0.20 \
  --max-dataset-ratio 0.50 \
  --relaxed-dataset-ratio 0.55 \
  --max-hdtf-ratio 0.15 \
  --max-voxceleb-ratio 0.05 \
  --max-per-source 1 \
  --random-seed 20260720
```

最终冻结器会再次执行过滤和去重，因此即使误传未去重 pool，也不会直接污染 split；正式命令仍应使用 `combined_pool_deduplicated.jsonl`。

## 3. 硬门槛

每条 test-main 必须满足：

```text
audio_only_pass = true
video_only_pass = true
full_av_pass = true
repeat review consensus = true（被抽中重复审核时）
每个 raw_source_id <= 1
每个 pair_group_id <= 1
inverse 不进入 test-main
train/val/test source、pair、inverse group 无交叉
```

Contextual speech 额外要求：

```text
speech_role in {contextual_speech, speech_with_event}
transcript_like = false
full_av_required = true
recomputed_asr_risk <= 0.35
video_context_strength >= 0.60
audio_delta_strength >= 0.70
audio_only_solvability < 0.85
```

ASR-only、generic talking head、transcript-like 和审核不确定样本进入 diagnostic，不计入主测试集。

## 4. 输出与验收

`benchmark_v1` 至少包含：

```text
test_main_150.jsonl
test_asr_diagnostic.jsonl
train.jsonl
val.jsonl
frozen_benchmark_manifest.json
frozen_benchmark.sha256
test_holdout_identities.json
review_agreement_summary.json
automatic_review_summary.json
rejection_breakdown.json
subtype_dataset_crosstab.json
split_summary.json
leakage_audit.json
asr_diagnostic_summary.json
benchmark_quality_report.md
```

验收必须同时满足：

```text
test_main_count = 150
test subtype = 90 / 30 / 30
duplicate_pair_count = 0
missing_media_count = 0
leakage.violation_count = 0
selection_uses_retrieval_model_scores = false
frozen SHA256 已生成
```

若任何一项失败，先报告缺口，不能手工补低质量样本。

## 5. 冻结后的实验规则

新 benchmark 会改变 train/val/test 的 source 身份。旧 adapter、旧 cache 和旧 test 指标不能作为新主表结果。冻结后必须：

```text
重新 prepare records
重新编码新的 train/val/test/gallery
只用 validation 选择训练步数和超参数
重新训练 adapter
在冻结 test-main 上运行多 seed 与七种 audio necessity 模式
```

旧实验只保留为 pilot 或方法开发证据。
