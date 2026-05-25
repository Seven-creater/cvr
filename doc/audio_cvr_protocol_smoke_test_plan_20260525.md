# Audio-CVR Protocol Smoke / Full Eval 执行说明

本文档把 Audio-CVR protocol 转成可执行流程。`1% smoke` 和后续全量评估复用同一套代码；区别只在参数规模，不在评估逻辑。

## 1. 目标

本流程不重新设计模型，也不新增 loss。它只验证：

1. gallery 构造是否正确；
2. reference/local/typed hard negatives 是否覆盖；
3. false-negative guard 是否生效；
4. `V+A+T vs V+T`、`A+T vs T-only-fullAV` 这类 audio necessity 消融能否规范跑；
5. Base E5 和当前 E5 recipe adapter 在不同 gallery 下的行为。

`random gallery` 只作为 sanity check。正式分析重点看 `reference`、`local_same_source`、`typed_hardneg` 和后续 `audio_necessity`。

## 2. 主要代码入口

通用后处理模块：

```bash
python3 -m app.audio_cvr_protocol_eval
```

兼容旧名称：

```bash
python3 -m app.audio_cvr_protocol_smoke
```

通用评估脚本：

```bash
bash scripts/run_audio_cvr_protocol_eval.sh
```

1% pilot 便捷脚本：

```bash
bash scripts/run_audio_cvr_protocol_smoke.sh
```

两者复用同一套 summarizer。`smoke.sh` 只是默认 `max-train-records=64`、`max-eval-records=30`；全量应使用 `run_audio_cvr_protocol_eval.sh`，并按需要把 `--max-train-records 0 --max-eval-records 0` 留为全量。

## 3. 输入检查

`<RUN_DIR>` 至少应包含：

```text
b_main_audio_cvr_triplets.jsonl
b_extended_audio_cvr_triplets.jsonl
b_diagnostic_audio_cvr_triplets.jsonl
b_all_audio_cvr_triplets.jsonl
audio_necessity_eval_manifest.json
benchmark_quality_summary.json
```

每条 B-main 记录重点检查：

```text
reference_video
target_video
edit_text
raw_source_id / source_id / source_disjoint_group_id
b_subtype / audio_delta_type
reference_satisfies_edit=false
target_satisfies_edit=true
audio_delta_strength
visual_shortcut_risk
asr_degeneracy_risk
audio_delta_hard_negatives
manual_review_required
```

每条 hard negative 重点检查：

```text
type / negative_type
video
source_id / raw_source_id
satisfies_edit=false
verification_accept=false
verification_status
temporal_relation
missing_reason
```

## 4. 产物

数据质量与人工复核：

```text
<OUTPUT_DIR>/data_quality_summary.json
<OUTPUT_DIR>/data_quality_summary.md
<OUTPUT_DIR>/human_review_cases.jsonl
<OUTPUT_DIR>/human_review_summary.md
```

评估汇总：

```text
<OUTPUT_DIR>/protocol_eval_summary.json
<OUTPUT_DIR>/gallery_protocol_results.md
<OUTPUT_DIR>/audio_necessity_results.md
<OUTPUT_DIR>/hard_negative_breakdown.md
<OUTPUT_DIR>/topk_errors.jsonl
<OUTPUT_DIR>/topk_errors.md
<OUTPUT_DIR>/advisor_brief.md
```

每个 eval 子目录仍保留原始诊断：

```text
summary.json
comparison.md
per_query_topk.jsonl
per_query_scores.jsonl
score_diagnostics.json
adapter_geometry.json
```

## 5. 1% 最小执行命令

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main

RUN_DIR=<RUN_DIR>
OUTPUT_DIR=<OUTPUT_DIR>
ADAPTER_DIR=<E5_CACHE_DIR>/adapter

bash scripts/run_audio_cvr_protocol_smoke.sh \
  --run-root "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --gallery-size <GALLERY_SIZE> \
  --seed <SEED> \
  --protocols random,reference,local_same_source,typed_hardneg \
  --video-audio-mode on
```

如果只先跑最关键协议，可以用：

```bash
--protocols reference,typed_hardneg
```

## 6. 全量复用命令形状

全量不换代码，只换参数：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main

RUN_DIR=<RUN_DIR>
OUTPUT_DIR=<OUTPUT_DIR>
ADAPTER_DIR=<E5_CACHE_DIR>/adapter

bash scripts/run_audio_cvr_protocol_eval.sh \
  --run-root "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --adapter-dir "$ADAPTER_DIR" \
  --run-label "Full Audio-CVR Protocol Eval" \
  --gallery-size <GALLERY_SIZE> \
  --seed <SEED> \
  --max-train-records 0 \
  --max-eval-records 0 \
  --protocols random,reference,local_same_source,typed_hardneg \
  --video-audio-mode on
```

全量不要把 `random` 当主结论。主报告优先使用 `reference`、`local_same_source`、`typed_hardneg`。

## 7. Audio Necessity 消融

当前 E5 cache 已支持 `--video-audio-mode on|off`：

```bash
python3 -m app.e5_audio_delta_train cache-embeddings \
  --records-dir <RECORDS_DIR> \
  --output-dir <CACHE_DIR> \
  --video-audio-mode off
```

`off` 用于 `V+T` / audio-off 诊断；`on` 用于 `V+A+T`。原则是：如果测试 audio-on，query 和 gallery 两侧都必须 audio-on；如果测试 audio-off，两侧都必须 audio-off。

当前 E5 cache 也支持 query 侧输入模式：

```bash
--query-input-mode composed   # reference video + edit_text，默认
--query-input-mode text_only  # edit_text only，用于 T-only-fullAV
--query-input-mode video_only # reference video only，用于 V+A / 纯相似 AV 诊断
```

因此今天可以先规范跑：

```text
T-only-fullAV: --query-input-mode text_only --video-audio-mode on
V+T:           --query-input-mode composed  --video-audio-mode off
V+A:           --query-input-mode video_only --video-audio-mode on
V+A+T:         --query-input-mode composed  --video-audio-mode on
```

`A-only` / `A+T` 需要严格 audio-only payload 或音频抽取后的 encoder 支持；不要用 full video payload 冒充 audio-only。该部分后续单独补，不影响先比较最关键的 `V+A+T vs V+T`。

## 8. 解释规则

如果 random 高但 reference/local 低：

```text
说明 random gallery 太简单，正式 benchmark 必须使用 reference/local/typed hard negatives。
```

如果 `V+A+T` 明显高于 `V+T`：

```text
说明 audio 在 video-text composed retrieval 基础上提供额外价值。
```

如果 `V+A+T` 和 `V+T` 接近：

```text
可能是样本仍可由视觉+文本解决，或模型没有有效利用 audio。优先检查 visual_hard、reference_negative、target-reference gap。
```

如果 `R@1` 低但 `R@5/R@10` 高：

```text
target 已进入前列，但经常被 reference/local hard negative 压住；这是方向性排序问题，不是完全检索失败。
```

## 9. 给导师的最短汇报模板

```text
本次 1% Audio-CVR protocol smoke 不追求最终性能，而是验证测试协议是否成立。
结果重点看三件事：
1. random gallery 是否虚高；
2. reference/local/typed hard negatives 是否显著提高难度；
3. audio-on 的 V+A+T 是否相比 audio-off 的 V+T 改善 target_beats_reference 和 target-reference score gap。

如果 random 高但 reference/local 低，这不是失败，而是说明 protocol 成功暴露了 Audio-CVR 的真实难点。
```
