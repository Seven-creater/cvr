# AVIGATE Agent Rerank 实验结果总结

## 实验配置

| 配置项 | 值 |
|--------|-----|
| 代码版本 | `aae5089` (parallelize omni descriptions and cache videos) |
| 测试集 | MSRVTT JSFUSION test (1000 samples) |
| 评估模式 | V2T (Video-to-Text) + T2V (Text-to-Video) |
| Omni Concurrency | 2 |
| Rerank Window | 5 |
| GPU 分配 | 4 shards × GPU 4-7 (AVIGATE) + GPU 2-3 (Omni) |

---

## 1. V2T (Video-to-Text) 结果

### 1.1 整体指标

| 指标 | Round1 (Baseline) | Final (Agent) | 变化 |
|------|-------------------|---------------|------|
| **R@1** | 0.435 | 0.454 | **+0.019** ✅ |
| **R@5** | 0.723 | 0.720 | -0.003 |
| **R@10** | 0.819 | 0.819 | 持平 |

### 1.2 与论文复现值对比

| 指标 | 论文复现值 | 实验 Round1 | 差异 |
|------|-----------|-------------|------|
| R@1 | 0.435 | 0.435 | ✅ 完全一致 |
| R@5 | 0.723 | 0.723 | ✅ 完全一致 |
| R@10 | 0.819 | 0.819 | ✅ 完全一致 |

### 1.3 Shard 分布详情

| Shard | Round1 R@1 | Final R@1 | 提升 |
|-------|-----------|-----------|------|
| shard_0 | 0.428 | 0.452 | +0.024 |
| shard_1 | 0.464 | 0.476 | +0.012 |
| shard_2 | 0.460 | 0.492 | +0.032 |
| shard_3 | 0.388 | 0.396 | +0.008 |

### 1.4 关键统计

- **avg_omni_calls**: 2.0
- **audio_off_rate**: 0.0
- **fallback_rate**: 0.005

---

## 2. T2V (Text-to-Video) 结果

### 2.1 整体指标

| 指标 | Round1 (Baseline) | Final (Agent) | 变化 |
|------|-------------------|---------------|------|
| **R@1** | 0.456 | 0.457 | +0.001 |
| **R@5** | 0.723 | 0.723 | 持平 |
| **R@10** | 0.821 | 0.821 | 持平 |

### 2.2 与论文复现值对比

| 指标 | 论文复现值 | 实验 Round1 | 差异 |
|------|-----------|-------------|------|
| R@1 | 0.464 | 0.456 | ⚠️ -0.008 |
| R@5 | 0.732 | 0.723 | ⚠️ -0.009 |
| R@10 | 0.827 | 0.821 | ⚠️ -0.006 |

### 2.3 Shard 分布详情

| Shard | Round1 R@1 | Final R@1 | 变化 |
|-------|-----------|-----------|------|
| shard_0 | 0.368 | 0.404 | +0.036 |
| shard_1 | 0.528 | 0.516 | -0.012 |
| shard_2 | 0.468 | 0.472 | +0.004 |
| shard_3 | 0.460 | 0.436 | -0.024 |

### 2.4 关键统计

- **avg_omni_calls**: 7.0
- **audio_off_rate**: 0.001
- **fallback_rate**: 0.02
- **query_rewrite_rate**: 0.822

---

## 3. 结论

### 3.1 V2T 结论

✅ **Agent 在 V2T 上有效**

- R@1 提升 **+1.9%** (0.435 → 0.454)，稳定且显著
- Round1 与论文复现值完全一致，验证跑法正确
- 4 个 shard 中 3 个有正向提升，1 个持平

### 3.2 T2V 结论

❌ **Agent 在 T2V 上几乎无效**

- R@1 仅提升 **+0.1%** (0.456 → 0.457)，可视为抖动
- Round1 与论文复现值有小幅偏差 (~0.8%)
- Query rewrite 发生率高 (82.2%)，但效果不明显
- 4 个 shard 中 2 个提升、2 个下降，相互抵消

### 3.3 可能原因分析

**T2V 效果不佳的可能原因：**

1. **Query Rewrite 质量问题**: 虽然 rewrite 发生率高，但 rewrite 后的 query 可能偏离原意或不够精准
2. **Rerank 策略局限**: 当前 rerank window=5 可能不足以捕获有效信息
3. **Omni 理解偏差**: Omni 对视频内容的理解与 AVIGATE 的 embedding 空间存在 gap
4. **Fallback 率较高**: 2% 的 fallback 可能损失了部分优化机会

---

## 4. 实验完整性验证

| 检查项 | 状态 |
|--------|------|
| git HEAD | `aae5089` ✅ |
| unittest | 19 tests OK ✅ |
| 8092 端口监听 | 正常 (vllm) ✅ |
| Omni 服务 | qwen2.5-omni 可用 ✅ |
| V2T 样本数 | 1000 ✅ |
| T2V 样本数 | 1000 ✅ |

---

## 5. 后续建议

### 5.1 V2T
- ✅ 当前策略已验证有效，可作为 baseline
- 可尝试调整 `rerank_window` 或 `omni_concurrency` 进一步优化

### 5.2 T2V
- 🔍 需要诊断 query rewrite 的具体质量
- 建议分析 rewrite 前后的 query 对比
- 考虑调整 T2V 的 prompt 策略或 rerank 逻辑
- 可尝试减小 `rerank_window` 或增加 `max_iter`

---

*实验完成时间: 2026-04-17*
*实验目录: `/data02/usr/wangqihao/Demo/test/cvr/runs/official_rerank_full/`*
