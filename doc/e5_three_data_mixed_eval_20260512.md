# E5-Omni 三数据集混合评测报告

- **日期**: 2026-05-12
- **commit**: `afe6816 Fix e5 audio-off video processing`
- **Gallery**: 1697 个视频（cvr_943: 943 + a_line: 340 + b_line: 414）
- **脚本**: `scripts/run_e5_cvr_eval.sh`
- **入口**: `app/e5_cvr_eval.py` + `app/e5_three_data_eval.py`
- **三种输入模式**：
  - V+T+A：video + text + audio（全模态基线）
  - V+T：video + text，双侧关闭音频
  - V+A：video + audio，去掉 edit_text

---

## 1. 数据集

三份数据混合评测，共 1697 条样本。

| 数据集 | 样本数 | 源视频 | 描述 |
|--------|-------|--------|------|
| cvr_943 | 943 | 193 个（94 daily_omni + 99 worldsense） | 原始 CVR 数据集 |
| a_line | 340 | - | Line-A 数据集 |
| b_line | 414 | - | Line-B 数据集 |

三元组格式同前：reference.mp4 + target.mp4 + edit_text.txt。

---

## 2. 方法

### 三种输入模式

| 模式 | query 输入 | 音频 | edit_text | target 音频 |
|------|-----------|------|-----------|------------|
| **V+T+A** | video + text | 开启（双侧） | 是 | 开启 |
| **V+T** | video + text | 关闭（双侧） | 是 | 关闭 |
| **V+A** | video only | 开启 | 否 | 开启 |

- V+T：`video_audio_mode=off`，query 和 target 都不提取音频
- V+A：`query_mode=video-only`，去掉 edit_text，只保留 reference video + 音频
- V+T+A 和 V+A 共用同一份 audio-on target index；V+T 单独编码无音频 target index

### 模型配置

| 参数 | 值 |
|------|-----|
| 模型 | e5-omni-7B |
| 架构 | Qwen2.5-Omni |
| Embedding 维度 | 3584 |
| torch_dtype | bfloat16 |
| video_max_pixels | 50176 |
| video_fps | 1 |
| batch_size | 1 |

---

## 3. 实验执行

- V+T+A：GPU 4，完整 target 编码 + query（先跑完）
- V+A：GPU 6，复用 V+T+A 的 target index，仅跑 query
- V+T：GPU 5，单独编码无音频 target + query（修复 `load_audio_from_video` bug 后补跑）

```bash
# V+T+A
bash scripts/run_e5_cvr_eval.sh \
  --gpu-id 4 --query-mode composed --video-audio-mode on

# V+A（复用 target index）
bash scripts/run_e5_cvr_eval.sh \
  --gpu-id 6 --query-mode video-only --video-audio-mode on \
  --target-index-dir runs/.../vta_audio_on/target_index

# V+T（无音频）
bash scripts/run_e5_cvr_eval.sh \
  --gpu-id 5 --query-mode composed --video-audio-mode off
```

最终汇总：
```bash
python3 -m app.e5_three_data_eval \
  --triplets-jsonl ... --run-root runs/e5_three_data_mixed_20260512_124128 \
  --topk 1,5,10
```

---

## 4. 结果

### 4.1 Overall

| 输入模式 | Query Count | R@1 | R@5 | R@10 |
|----------|------------:|----:|----:|-----:|
| V + T + A | 1697 | 0.1220 | 0.5162 | 0.8438 |
| **V + T** | **1697** | **0.1296** | **0.5286** | 0.8427 |
| V + A | 1697 | 0.0931 | 0.4349 | 0.7684 |

### 4.2 按数据集分表

| 输入模式 | Dataset | Count | R@1 | R@5 | R@10 |
|----------|---------|------:|----:|----:|-----:|
| V + T + A | cvr_943 | 943 | 0.1718 | 0.6278 | 0.9109 |
| V + T + A | a_line | 340 | 0.0647 | 0.4176 | 0.8029 |
| V + T + A | b_line | 414 | 0.0556 | 0.3430 | 0.7246 |
| **V + T** | **cvr_943** | **943** | **0.1845** | **0.6469** | **0.9226** |
| V + T | a_line | 340 | 0.0618 | 0.4088 | 0.7912 |
| V + T | b_line | 414 | 0.0604 | 0.3575 | 0.7029 |
| V + A | cvr_943 | 943 | 0.1294 | 0.5154 | 0.8388 |
| V + A | a_line | 340 | 0.0529 | 0.3559 | 0.7353 |
| V + A | b_line | 414 | 0.0435 | 0.3164 | 0.6353 |

### 4.3 R@1 主表

| 输入模式 | cvr_943 | a_line | b_line | Overall |
|----------|--------:|-------:|-------:|--------:|
| V + T + A | 0.1718 | 0.0647 | 0.0556 | 0.1220 |
| **V + T** | **0.1845** | 0.0618 | **0.0604** | **0.1296** |
| V + A | 0.1294 | 0.0529 | 0.0435 | 0.0931 |

---

## 5. 分析

### 5.1 V+T（无音频）整体最优

V+T 在 Overall R@1（0.1296 vs 0.1220）和 R@5（0.5286 vs 0.5162）上均优于 V+T+A，尤其在 cvr_943 上差距最明显（R@1 +1.3pp，R@5 +1.9pp）。这进一步验证了 943 单数据集实验的结论：**音频对当前 CVR 任务无正向贡献，反而引入噪声**。

### 5.2 edit_text 贡献显著且一致

| 对比 | R@1 差值 | R@5 差值 | R@10 差值 |
|------|---------:|---------:|----------:|
| V+T+A vs V+A | **+2.9pp** | **+8.1pp** | **+7.5pp** |
| V+T vs V+A | **+3.7pp** | **+9.4pp** | **+7.4pp** |

edit_text 在三个数据集上均带来稳定提升，是区分同源视频的关键信号。

### 5.3 三个数据集难度差异

- **cvr_943**：整体最好（V+T R@1=18.45%），源视频少（193个）但配对质量高
- **a_line**：居中（V+T R@1=6.18%）
- **b_line**：最难（V+T R@1=6.04%）

### 5.4 音频消融结论一致

与 943 单数据集实验（commit `09a826b`）结论一致：去掉音频后检索不降反升。两个独立实验交叉验证了音频在当前 CVR 任务中的负面影响。

| 实验 | 有音频 R@1 | 无音频 R@1 | 差值 |
|------|----------:|----------:|-----:|
| 943 单数据集（composed vs ref-silent） | 0.2025 | 0.2078 | +0.5pp |
| 三数据集混合（V+T+A vs V+T） | 0.1220 | 0.1296 | +0.8pp |

---

## 6. 输出文件

```
runs/e5_three_data_mixed_20260512_124128/
├── vta_audio_on/          # V+T+A
│   ├── target_index/      # 含音频的 target embeddings (1697×3584)
│   ├── smoke20/
│   └── full1697/
├── va_video_only_audio_on/ # V+A（复用 vta 的 target index）
│   ├── smoke20/
│   └── full1697/
├── vt_audio_off/          # V+T（无音频，单独编码 target）
│   ├── target_index/      # 无音频 target embeddings (1697×3584)
│   ├── smoke20/
│   └── full1697/
└── comparison_by_dataset.md
```
