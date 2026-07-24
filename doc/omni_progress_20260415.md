# Omni Checker Progress Report (2026-04-15)

## 1. 当前目标

当前项目的主线是重建一个极简、可训练的双向检索 agent 系统，面向 `MSRVTT` 数据集。

现阶段优先级不是整套 agent 批量评测，而是先把最关键的运行链路打通：

1. 冻结检索器
2. 通过 `Qwen2.5-Omni` 对候选视频/文本做运行时检查
3. 保证 checker 的输入输出格式稳定、可复用
4. 在此基础上再逐步扩展到 `T2V / V2T` agent-case 和 batch

本次阶段汇报聚焦于：

- `OpenAIOmniChecker -> Omni 服务 -> 本地视频` 这条链是否真正跑通
- 模型是否能稳定输出完整结构化 JSON

---

## 2. 当前代码状态

当前仓库关键提交为：

- `33e6b25` `align omni checker with official video_url client`
- `02d5a74` `harden omni checker json handling`
- `88430c2` `strengthen omni checker schema prompts`

其中 `88430c2` 是本次验证使用的最新版本。

本次与 Omni 相关的核心改动集中在：

- `app/omni_checker.py`
- `tests/test_omni_checker.py`

### 2.1 已完成的关键修正

1. 请求格式对齐官方成功范式  
   不再使用错误的 `type="video"` / `type="file"`，而是统一使用 `video_url`

2. 本地视频自动转为 base64 data URL  
   参考官方 example client，本地 `mp4` 在请求前自动编码，不再直接把 `file://...` 原样塞给接口

3. 增强结构化输出约束  
   在 system prompt 中明确要求：
   - 必须返回 8 个字段
   - `{"is_match": true}` 这种半截 JSON 无效
   - 给出完整示例 JSON 让模型模仿

4. 增强健壮性  
   即使模型偶尔返回不完整或不规范 JSON，checker 也不会直接崩溃；同时会显式记录缺失字段，避免误判为“结构化输出已经很好”

### 2.2 本地测试状态

当前本地全量测试通过：

- `14 tests OK`

说明：

- checker 请求格式更新后没有破坏现有 agent loop / retriever 逻辑
- 新增的 JSON 容错和缺字段标记逻辑已被测试覆盖

---

## 3. 服务器端验证结果

### 3.1 服务环境

服务器已成功拉到：

- `88430c2` `strengthen omni checker schema prompts`

Omni 服务当前状态：

- 环境：`omni_src`
- 服务端口：`127.0.0.1:8092`
- 监听正常
- `/v1/models` 返回正常

服务最终成功启动的关键条件是：

- 清理旧的 `omni` 环境残留服务
- 显式使用空闲 GPU `2,3`
- 将 `--gpu-memory-utilization` 降到 `0.70`

### 3.2 最小链路验证

服务器已成功执行最小 `OpenAIOmniChecker` 调用，输入为：

- query: `Describe the main events in this video.`
- video: `video554.mp4`

调用结果如下：

```python
{
    "is_match": True,
    "confidence": 1.0,
    "visual_match": 0.76,
    "audio_match": 0.45,
    "main_events": [
        "woman pours milk into pot",
        "woman cuts vanilla pod"
    ],
    "missing_elements": [],
    "reason": "The visible cooking actions match the query.",
    "rewrite_suggestion": ""
}
```

这是当前最重要的阶段性成果：

**`OpenAIOmniChecker -> Omni 服务 -> 本地视频 -> 完整 8 字段 JSON` 已经跑通。**

### 3.3 原始响应确认

服务器进一步抓取了原始 `chat/completions` 返回，模型真实输出为：

```json
{
  "is_match": true,
  "confidence": 1.0,
  "visual_match": 0.76,
  "audio_match": 0.45,
  "main_events": [
    "woman pours milk into pot",
    "woman cuts vanilla pod"
  ],
  "missing_elements": [],
  "reason": "The visible cooking actions match the query.",
  "rewrite_suggestion": ""
}
```

这说明当前成功并不是“本地 fallback 硬补字段”，而是：

**模型本身已经开始按新 prompt 返回完整 schema。**

---

## 4. 本次问题是如何被解决的

本次排查过程里，真正起作用的不是单一修改，而是下面三步连起来：

### 4.1 服务链路打通

前期主要障碍包括：

- `vllm-omni` 安装版本不兼容
- 普通 `vllm` 和 `vllm-omni` 环境混装
- 旧服务残留占用 GPU 显存
- 错误的多模态请求格式

经过多轮排查后，最终确认：

- `omni_src` 环境可用
- 新服务能稳定运行在 `8092`
- 官方 example client 能成功处理 `video554.mp4`

### 4.2 请求格式对齐官方范式

最终验证表明，正确的路径是：

- `video_url`
- 本地视频编码为 `data:video/mp4;base64,...`

而不是：

- `type="video"`
- `type="file"`
- 或直接塞原始 `file://...`

### 4.3 Prompt 约束加强

此前模型一直只返回：

```json
{"is_match": true}
```

这说明问题不在链路，而在输出约束不够强。

本次对 prompt 的有效增强包括：

1. 明确写出 “All 8 keys are mandatory”
2. 明确声明 `{"is_match": true}` 是无效答案
3. 提供完整 JSON 示例
4. 在 user prompt 中再次强调“不返回完整 8 字段就算错误”

从服务器结果看，这一步已经奏效。

---

## 5. 当前已经确认的结论

### 已确认成立

1. `omni_src` 环境可以跑通 `Qwen2.5-Omni`
2. `OpenAIOmniChecker` 可以稳定访问 8092 服务
3. 本地视频输入链路可用
4. 模型现在可以输出完整 8 字段结构化 JSON
5. checker 当前已经足够进入下一阶段的小规模 agent-case 验证

### 还没有完成

1. 真实 `feature_dir` 仍未定位到
2. 基于真实 frozen feature 的正式 baseline 还没跑
3. `T2V / V2T` 的完整 agent-case 还没正式用真实特征做端到端验证
4. 还没有进入 batch 评测和 RL 数据导出阶段

---

## 6. 当前阶段性判断

到目前为止，项目已经跨过了最容易卡死的一关：

**Omni checker 不再只是“概念上可用”，而是已经在服务器上对真实视频完成了端到端结构化推理。**

这意味着下一步我们不需要再停留在：

- 装环境
- 猜请求格式
- 调服务
- 猜 JSON schema

而可以开始进入真正和方法相关的工作：

- 单条 `T2V agent-case`
- 单条 `V2T agent-case`
- 再逐步扩到 batch

---

## 7. 建议的下一步

建议按下面顺序推进：

1. 保持当前 8092 服务不动
2. 用当前仓库代码做一个真实 `T2V agent-case`
3. 再做一个真实 `V2T agent-case`
4. 如果链路稳定，再开始找真实 `feature_dir`
5. 最后再进入批量评测和 RL 数据样本导出

当前最合理的策略不是继续折腾基础设施，而是：

**在已经打通的 Omni checker 上，开始推进最小 agent 闭环。**

