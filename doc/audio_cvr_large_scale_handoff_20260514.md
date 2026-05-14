# Audio-CVR 6-9s 大规模 B 线构造交接文档

更新时间：2026-05-14

## 1. 运行目标

本次只构造大规模 B 线 Audio-CVR 数据集。

- 输入：服务器上已经准备好的全模态原始视频数据集。
- 切片：6-9 秒，默认 8 秒，输出到新目录 `clips/audio_cvr_6_9s/`。
- 构造方法：最新 B 线 `b_audio_blind_review_v2`。
- 输出：B 线 triplets、manual review bundle、annotation/proposal cache、summary。
- 不跑 A 线，不跑 e5，不跑 AVIGATE，不跑 agent，不修改旧 943 数据。

## 2. 固定服务器目录

```bash
REPO=/data02/usr/wangqihao/Demo/test/cvr_clean_main
DATA_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
RAW_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw
CLIP_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/audio_cvr_6_9s
MODEL_PATH=/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
```

运行输出默认写到：

```bash
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/audio_cvr_bline_6_9s_full_<timestamp>
```

## 3. 数据集清单

| 数据集 | 视频路径 | 规模和时长 | 本次处理规则 |
|---|---|---:|---|
| `daily_omni` | `raw/daily_omni/video/` | 1,196 mp4，约 30s | 重新切 8s |
| `worldsense` | `raw/worldsense/videos/` | 1,662 mp4，30-540s | 重新切 8s |
| `hdtf` | `raw/hdtf/videos/` | 400 长视频，30-140s | 只用 `videos/`，不要用 `clips/` |
| `avatar` | `raw/avatar/` 和 `raw/avatar/video/` | 10,000 mp4，约 10s | 8s + tail clip，保证短视频可成 pair |
| `vggsound` | `raw/vggsound/scratch/` | 20,000 mp4，约 10-15s | 8s + tail clip |
| `vgg_monoaudio` | `raw/vgg_monoaudio/inter_class/mixed/` | 1,071 mp4，约 8s | 有视频和音轨才使用 |
| `voxceleb` | `raw/voxceleb/vox2_mp4/dev/` | 1,092,009 mp4，约 4-9s，224×224 | 同一父目录短 mp4 聚成 single-source group；跳过 `vox1/` 和 `vox2_aac/` |

注意：`raw/hdtf/clips/` 大多约 3.2s，低于本次 6s 下限，不能作为主 B 线切片输入。

## 4. 模型和环境要求

必须使用 Qwen3-Omni：

```bash
/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
```

vLLM OpenAI API 配置：

- port: `8093`
- served model name: `qwen3-omni-30b-a3b-instruct`
- GPU: `0,1,2,3`
- tensor parallel: `4`
- max model len: `16384`
- max num seqs: `8`
- dtype: `bfloat16`

运行前检查：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

which python3
which ffmpeg
which ffprobe
python3 -c "import torch, vllm; print('ok')"
test -f /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct/config.json
```

## 5. 只读数据检查

服务器执行人员先运行以下命令，只检查，不改代码：

```bash
cd /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval

test -d raw/daily_omni/video
test -d raw/worldsense/videos
test -d raw/hdtf/videos
test -d raw/avatar
test -d raw/vggsound/scratch
test -d raw/vgg_monoaudio/inter_class/mixed
test -d raw/voxceleb/vox2_mp4/dev
find raw/voxceleb/vox2_mp4/dev -name "*.mp4" | head
```

如果某一项不存在，停止，把缺失路径回传，不要现场改代码。

## 6. 全量后台运行命令

服务器执行人员只运行命令，不改代码。

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
git pull --ff-only origin main

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

mkdir -p logs

RUN_ROOT=/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/audio_cvr_bline_6_9s_full_$(date +%Y%m%d_%H%M%S)
LOG=logs/audio_cvr_bline_6_9s_full_$(date +%Y%m%d_%H%M%S).log

setsid nohup bash scripts/run_audio_cvr_bline_6_9s_full_4gpu.sh \
  --run-root "$RUN_ROOT" \
  --start-omni auto \
  --gpu-ids 0,1,2,3 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --clip-seconds 8 \
  --min-clip-seconds 6 \
  --max-clip-seconds 9 \
  --propose-shards 64 \
  --propose-parallel-jobs 8 \
  --concurrency 4 \
  --request-timeout-seconds 240 \
  --shard-timeout-seconds 10800 \
  --target-b-count 1000000 \
  > "$LOG" 2>&1 < /dev/null &

echo $! | tee logs/audio_cvr_bline_6_9s_full.pid
echo "$RUN_ROOT"
echo "$LOG"
```

VoxCeleb 注意事项：

- 脚本只扫描 `raw/voxceleb/vox2_mp4/dev/`。
- `raw/voxceleb/vox1/` 是 wav/txt，不进入主 B 线。
- `raw/voxceleb/vox2_aac/` 是纯音频，不进入主 B 线。
- VoxCeleb mp4 多数只有 4-9s；脚本会把 6-9s 的短 mp4 按父目录聚成 single-source group，并用 hardlink/copy 写入 clip cache，避免对 100 万级短 mp4 重编码。

## 7. 缓存和断点续跑

不要删除 run 目录，不要删除 clip 目录。

- 切片：每个 mp4 先写临时文件，成功后原子替换成最终文件；重跑会复用已完成 mp4。
- VoxCeleb 短 mp4：完整 6-9s mp4 使用 hardlink/copy 缓存，不走 ffmpeg 重编码。
- annotation：每条写入 `single_source_annotations.jsonl`，中断后按 `clip_id` 复用。
- propose：每条写入 `accepted_progress_*.jsonl` / `rejected_progress_*.jsonl`。
- merge：可以从 ranked/progress JSONL 重新生成最终 summary 和 review bundle。

如果 vLLM 假死：

1. 停掉当前构造脚本。
2. 重启 8093 Qwen3-Omni 服务，或让脚本用 `--start-omni auto` 自检重启。
3. 使用同一个 `RUN_ROOT` 重跑，不要换目录。

## 8. 进度检查命令

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
PID=$(cat logs/audio_cvr_bline_6_9s_full.pid)
ps -p "$PID" -o pid,pgid,stat,etime,cmd || true

LOG=$(ls -t logs/audio_cvr_bline_6_9s_full_*.log | head -1)
tail -100 "$LOG"

RUN_ROOT=$(ls -td runs/audio_cvr_bline_6_9s_full_* | head -1)
echo "$RUN_ROOT"

cat /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/audio_cvr_6_9s/_manifests/audio_cvr_6_9s_summary.json 2>/dev/null || true
wc -l "$RUN_ROOT/single_source_annotations.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/b_candidates.jsonl" 2>/dev/null || true
cat "$RUN_ROOT"/b_shards/accepted_progress_*.jsonl 2>/dev/null | wc -l
cat "$RUN_ROOT"/b_shards/rejected_progress_*.jsonl 2>/dev/null | wc -l
nvidia-smi -i 0,1,2,3
```

## 9. 最终验收命令

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
RUN_ROOT=$(ls -td runs/audio_cvr_bline_6_9s_full_* | head -1)
echo "$RUN_ROOT"

cat "$RUN_ROOT/summary.json"
wc -l "$RUN_ROOT/b_speech_audio_content_triplets.jsonl"
ls "$RUN_ROOT/manual_review/B" | head
find "$RUN_ROOT/manual_review/B" -maxdepth 2 -type f | head -30

cat /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/audio_cvr_6_9s/_manifests/audio_cvr_6_9s_summary.json
```

验收重点：

- `b_speech_audio_content_triplets.jsonl` 存在且有样本。
- `manual_review/B/` 有可审查样本。
- 日志没有持续大量 `Input length exceeds`、`fallback_pair_proposal`、`timeout`。
- summary 中 B 线使用的是 `b_audio_blind_review_v2`。

## 10. 禁止事项

服务器执行人员必须遵守：

- 不要改代码。
- 不要改原始数据。
- 不要覆盖 `clips/audio_cvr_8_12s/`。
- 不要删除 `clips/audio_cvr_6_9s/` 或当前 `RUN_ROOT`。
- 不要跑 A 线。
- 不要跑 e5、AVIGATE、agent。
- 不要启动 VACE 或任何视频生成模型。
- 不要把纯 ASR / 纯音频数据混入主 B 线。

## 11. 失败时回传信息

如果失败，回传以下信息，不要现场 patch：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
git rev-parse --short HEAD
tail -200 logs/audio_cvr_bline_6_9s_full_*.log
RUN_ROOT=$(ls -td runs/audio_cvr_bline_6_9s_full_* | head -1)
cat "$RUN_ROOT/summary.json" 2>/dev/null || true
wc -l "$RUN_ROOT/single_source_annotations.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/b_candidates.jsonl" 2>/dev/null || true
cat "$RUN_ROOT"/b_shards/accepted_progress_*.jsonl 2>/dev/null | wc -l
cat "$RUN_ROOT"/b_shards/rejected_progress_*.jsonl 2>/dev/null | wc -l
curl -sS http://127.0.0.1:8093/v1/models
nvidia-smi -i 0,1,2,3
```
