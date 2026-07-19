# Audio-CVR AAAI 一周执行清单

## 1. 当前阶段

服务器正在构造新的候选池，目标是冻结约 150 条正式测试 query。构造期间不读取旧 test 结果挑选新样本，也不继续增加训练 loss。当前并行完成三项与数据规模无关的工作：

1. 把新测试集的人审、去泄漏、冻结和哈希做成可复用命令；
2. 仅在旧 validation cache 上补充训练步数选择，不接触 test；
3. 预注册多组成对比较和统计输出。

## 2. 正式测试集冻结

### 2.1 生成盲审池

新 run 完成后，从 `B-main` 正向样本中排除旧 train/val/test 的 source、pair 和 sample。建议先准备 225 条，给人工拒绝和类别配额留余量：

```bash
python3 -m app.audio_cvr_paper_experiment prepare-benchmark-review \
  --input-path <NEW_RUN>/b_main_audio_cvr_triplets.jsonl \
  --exclude-path <OLD_SPLITS>/train.jsonl \
  --exclude-path <OLD_SPLITS>/val.jsonl \
  --exclude-path <OLD_SPLITS>/test_main.jsonl \
  --output-dir <OUTPUT_DIR>/benchmark_review \
  --review-count 225 \
  --repeat-review-fraction 0.20 \
  --random-seed 20260719
```

该命令不读取模型分数。它输出第一轮全量盲审表和 20% 重复复核表。

### 2.2 最小人审标准

每条样本只有以下五项均为 `true` 且 `decision=passed` 才能进入候选测试集：

```text
edit_audio_only
reference_does_not_satisfy_edit
target_satisfies_edit
video_only_cannot_identify_target
hard_negatives_do_not_satisfy_edit
```

`failed` 和 `uncertain` 均不进入正式 test。重复复核发生分歧的样本同样排除。这里的人审只决定数据是否有效，不看 Base E5 或 adapter 是否答对。

### 2.3 冻结 150 条

```bash
python3 -m app.audio_cvr_paper_experiment finalize-benchmark \
  --candidate-path <OUTPUT_DIR>/benchmark_review/benchmark_review_candidates.jsonl \
  --review-path <OUTPUT_DIR>/benchmark_review/human_review_round1_completed.jsonl \
  --review-path <OUTPUT_DIR>/benchmark_review/human_review_round2_completed.jsonl \
  --exclude-path <OLD_SPLITS>/train.jsonl \
  --exclude-path <OLD_SPLITS>/val.jsonl \
  --exclude-path <OLD_SPLITS>/test_main.jsonl \
  --output-dir <OUTPUT_DIR>/frozen_test \
  --target-count 150 \
  --minimum-count 100 \
  --max-speech-ratio 0.35 \
  --max-dataset-ratio 0.60 \
  --min-strict-local-coverage 0.50 \
  --random-seed 20260719
```

正式目标是 150 条；100 是论文实验仍可运行的下限，不是推荐规模。如果 strict local coverage 暂时无法达到 50%，先修数据而不是把跨源 fallback 冒充 strict local。

冻结产物：

```text
test_main.jsonl
frozen_benchmark_manifest.json
frozen_benchmark.sha256
test_holdout_identities.json
```

manifest 明确记录数据来源、subtype、人审一致率、local coverage、旧 split 泄漏和是否使用模型分数。后续构建 train/val 时还必须读取 `test_holdout_identities.json`，排除其中全部 source、pair 和 sample。

## 3. 构造期间可跑的 Validation-only 实验

当前 1000 steps 是旧 validation 搜索的上边界。为确认是否需要更晚停止，仅复用旧 V+A+T train/validation cache，搜索：

```text
steps = 700, 1000, 1300, 1600
lr = 1e-3
batch = 8
seeds = 13, 23, 42
```

启动形状：

```bash
setsid nohup bash scripts/run_audio_cvr_validation_extension.sh \
  --cache-dir <TRAIN_VAL_CACHE> \
  --output-dir <OUTPUT_DIR>/validation_extension \
  --gpu-ids 0,1,2,3,4,5 \
  --steps-grid 700,1000,1300,1600 \
  --seeds 13,23,42 \
  --learning-rate 0.001 \
  --batch-size 8 \
  > <OUTPUT_DIR>/validation_extension.log 2>&1 < /dev/null &
```

选择规则为 one-standard-error rule：先找到 validation R@1 最优均值，再在其一个标准误范围内选择训练步数最少的配置。它比直接选择最高点更抗小 validation 集波动。脚本不编码视频、不读取 test，也不会启动或杀死 vLLM。

## 4. 最终统计比较

新 150-query test 冻结后，模型配置和五个 final seeds 固定。除主比较 `V+A+T - V+T` 外，预注册：

```text
V+A+T - V+A        edit text 的增量
V+A+T - V-only     audio+text 相对纯视觉
V+T - V-only       edit text 在无音频条件下的增量
A+T - A-only       edit text 在音频条件下的增量
```

`aggregate-final` 对每组输出 query-paired bootstrap CI、sign-flip randomization p 和跨比较 Holm 校正。主结论仍只以 `V+A+T - V+T` 为 primary endpoint，其他比较用于解释模态贡献，避免事后挑显著结果。

## 5. 论文可写条件

新结果进入主表前必须同时满足：

1. test 为 B-main、正向、人工通过且与旧 train/val/test source-disjoint；
2. 测试集及 gallery manifest 已冻结并有 SHA-256；
3. 训练步数只由 validation 决定；
4. 五个 final seeds 使用同一配置、同一 1000-item gallery；
5. 所有模态模式共享相同 sample IDs、positive index 和 reference index；
6. loss curve 与 summary 无 NaN/Inf；
7. 主比较报告 effect size、95% CI 和 p-value，不只报告单个 R@1；
8. strict local 与 fallback 分开报告。

## 6. 一周优先级

| 优先级 | 工作 | 截止标准 |
|---|---|---|
| P0 | 新候选构造与 150 条盲审 | 冻结 manifest、hash、零泄漏 |
| P0 | validation-only 选步 | one-SE 规则产生唯一配置 |
| P0 | 新 test 五 seed 七模式 | 完整 mean/std 与 paired statistics |
| P1 | strict local / reference 错例分析 | 分项指标与典型 top-k 案例 |
| P1 | 数据统计和人审一致率 | 可直接进入 dataset section |
| P2 | 外部 baseline | 只能使用严格对齐且能复现的实现 |

一周内不再新增 loss、不启用 LoRA、不用 test 调超参数，也不把随机 gallery 的高分当主证据。
