# Audio-CVR A+B 6-9s 大规模构造交接文档

更新日期：2026-05-16

## 1. 运行目标

这份文档面向一台新的大服务器。假设新服务器一开始没有代码、没有数据集、没有模型，需要从零准备环境，然后全量跑 **A-line + B-line Audio-CVR** 数据集构造。

本阶段目标：

- 全量构造 A-line：`visual_audio_anchor`。
- 全量构造 B-line：`speech_audio_content`，使用最新 `b_audio_blind_review_v2`。
- 统一切片：6-9 秒，默认 8 秒。
- 输入：7 个原始全模态视频数据集。
- 模型：Qwen3-Omni-30B-A3B-Instruct，通过 vLLM OpenAI API 提供服务。
- 输出：A/B triplets、B-line tier 分层、manual review bundle、annotation/proposal cache、summary、B-line split 文件。

本阶段不做：

- 不跑 e5 训练或评测。
- 不跑 AVIGATE。
- 不跑 agent。
- 不跑 VACE 或任何视频生成模型。
- 不修改原始数据。
- 不把纯 ASR / 纯音频数据混入 `B-main`。

## 2. 新服务器固定目录

建议新服务器使用以下目录，方便脚本、日志和后续交接一致：

```bash
REPO=/data02/usr/wangqihao/Demo/test/cvr_clean_main
DATA_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval
RAW_DATASETS_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets
RAW_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw
CLIP_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/audio_cvr_6_9s
MODEL_ROOT=/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone
QWEN3_OMNI_DIR=/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
```

全量运行输出默认写到：

```bash
/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/audio_cvr_ab_6_9s_full_<timestamp>
```

## 3. 系统环境

### 3.1 系统工具

必须安装：

- `git`
- `git-lfs`
- `ffmpeg`
- `ffprobe`
- `unzip`
- `tar`
- `rsync`
- `curl`
- `aria2c`，推荐

检查：

```bash
which git
which git-lfs
which ffmpeg
which ffprobe
which unzip
which tar
which rsync
which curl
```

### 3.2 Conda 环境

建议环境名为 `omni_src`：

```bash
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda create -n omni_src python=3.10 -y
conda activate omni_src

python -m pip install -U pip setuptools wheel
python -m pip install -U modelscope huggingface_hub hf_transfer
python -m pip install -U numpy pandas tqdm requests pydantic
python -m pip install -U torch torchvision torchaudio
python -m pip install -U vllm transformers accelerate qwen-omni-utils
```

检查：

```bash
python3 -c "import torch, vllm, transformers; print(torch.__version__); print('ok')"
```

### 3.3 代码仓库

推荐路径：

```bash
mkdir -p /data02/usr/wangqihao/Demo/test
cd /data02/usr/wangqihao/Demo/test
git clone https://github.com/Seven-creater/cvr.git cvr_clean_main
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
```

如果新服务器无法访问 GitHub，则由本地或旧服务器打包代码上传到同一路径。服务器执行人员只负责放置代码和运行命令，不现场修改代码。

检查：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
git rev-parse --short HEAD
test -f scripts/build_audio_cvr_6_9s_clips.sh
test -f scripts/run_audio_cvr_ab_6_9s_full_4gpu.sh
test -f scripts/run_audio_lines_single_source_reuse.sh
```

## 4. 模型下载

本阶段必须下载 Qwen3-Omni：

```bash
Qwen/Qwen3-Omni-30B-A3B-Instruct
```

ModelScope 下载命令：

```bash
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

mkdir -p /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone
cd /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone

setsid nohup bash -lc '
set -euo pipefail
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src
python -m pip install -U modelscope
modelscope download \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local_dir /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
test -f /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct/config.json
' > /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/download_qwen3_omni.log 2>&1 < /dev/null &
```

验收：

```bash
QWEN3_OMNI_DIR=/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct
test -f "$QWEN3_OMNI_DIR/config.json"
du -sh "$QWEN3_OMNI_DIR"
find "$QWEN3_OMNI_DIR" -maxdepth 1 -type f | sort | head -50
```

说明：

- A/B 数据集构造只需要 Qwen3-Omni。
- e5-omni 暂时不是本阶段必需模型。
- Qwen2.5-Omni 也不是本阶段必需模型。

## 5. raw_datasets 下载清单

服务器不能翻墙。下载优先级：

1. ModelScope。
2. hf-mirror。
3. 不直接访问 HuggingFace 原站。

下载包统一放到：

```bash
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets
```

解压后的可用数据统一整理到：

```bash
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw
```

### 5.1 VoxCeleb

下载源：

- ModelScope 推荐：<https://modelscope.cn/datasets/juliuscn/voxceleb>
- HF-Mirror 备用：<https://hf-mirror.com/datasets/ProgramComputer/voxceleb>

下载：

```bash
mkdir -p /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/VoxCeleb
cd /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/VoxCeleb

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src
python -m pip install -U modelscope

setsid nohup modelscope download \
  --dataset juliuscn/voxceleb \
  --local_dir /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/VoxCeleb \
  > download_voxceleb_modelscope.log 2>&1 < /dev/null &
```

目标结构：

```bash
raw/voxceleb/vox2_mp4/dev/
raw/voxceleb/vox2_aac/dev/
raw/voxceleb/vox1/
```

项目主扫描路径只使用：

```bash
raw/voxceleb/vox2_mp4/dev/
```

注意：VoxCeleb 默认高 ASR-risk，不直接进入 `B-main`。只有强视频语境、低 ASR 风险、无视觉捷径的样本才能进入主 benchmark。

### 5.2 WorldSense

下载源：

- ModelScope 推荐：<https://modelscope.cn/datasets/lmms-lab/WorldSense>
- HF-Mirror 备用：<https://hf-mirror.com/datasets/lmms-lab/WorldSense>

下载：

```bash
mkdir -p /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/worldsense
cd /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/worldsense

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src
python -m pip install -U modelscope

setsid nohup modelscope download \
  --dataset lmms-lab/WorldSense \
  --local_dir /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/worldsense \
  > download_worldsense_modelscope.log 2>&1 < /dev/null &
```

目标结构：

```bash
raw/worldsense/videos/
raw/worldsense/audios/
raw/worldsense/subtitles/
```

脚本扫描：

```bash
raw/worldsense/videos/
```

### 5.3 AVATAR

下载源：

- ModelScope 未找到。
- HF-Mirror 推荐：<https://hf-mirror.com/datasets/mipal/AVATAR>
- HuggingFace 原始地址：<https://huggingface.co/datasets/mipal/AVATAR>

下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/avatar

setsid nohup huggingface-cli download \
  --repo-type dataset mipal/AVATAR \
  --local-dir /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/avatar \
  > /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/avatar/download_avatar_hfmirror.log 2>&1 < /dev/null &
```

目标结构：

```bash
raw/avatar/
raw/avatar/video/
```

脚本扫描：

```bash
raw/avatar/
raw/avatar/video/
```

### 5.4 HDTF

下载源：

- ModelScope 未找到直接数据集。
- HF-Mirror 推荐：<https://hf-mirror.com/datasets/global-optima-research/HDTF>
- HuggingFace 原始地址：<https://huggingface.co/datasets/global-optima-research/HDTF>
- 原始 GitHub：<https://github.com/MRzzm/HDTF>

下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/hdtf

setsid nohup huggingface-cli download \
  --repo-type dataset global-optima-research/HDTF \
  --local-dir /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/hdtf \
  > /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/hdtf/download_hdtf_hfmirror.log 2>&1 < /dev/null &
```

目标结构：

```bash
raw/hdtf/videos/
raw/hdtf/clips/
```

脚本只扫描：

```bash
raw/hdtf/videos/
```

不要把 `raw/hdtf/clips/` 作为主输入，因为这些 clips 多数约 3.2 秒，低于本轮 6 秒下限。

### 5.5 VGG-MonoAudio

下载源：

- ModelScope 未找到。
- HF-Mirror 推荐：<https://hf-mirror.com/datasets/jnwnlee/vgg-monoaudio>
- HuggingFace 原始地址：<https://huggingface.co/datasets/jnwnlee/vgg-monoaudio>

下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/vgg_monoaudio

setsid nohup huggingface-cli download \
  --repo-type dataset jnwnlee/vgg-monoaudio \
  --local-dir /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/vgg_monoaudio \
  > /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/vgg_monoaudio/download_vgg_monoaudio_hfmirror.log 2>&1 < /dev/null &
```

目标结构：

```bash
raw/vgg_monoaudio/inter_class/mixed/
raw/vgg_monoaudio/inter_class/target_audio/
raw/vgg_monoaudio/intra_class/mixed/
raw/vgg_monoaudio/intra_class/target_audio/
```

脚本扫描：

```bash
raw/vgg_monoaudio/inter_class/mixed/
```

### 5.6 VGGSound

下载源：

- ModelScope 未找到直接托管。
- HF-Mirror 推荐：<https://hf-mirror.com/datasets/Loie/VGGSound>
- HuggingFace 原始地址：<https://huggingface.co/datasets/Loie/VGGSound>
- 官方主页：<https://www.robots.ox.ac.uk/~vgg/data/vggsound/>

下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/vggsound

setsid nohup huggingface-cli download \
  --repo-type dataset Loie/VGGSound \
  --local-dir /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/vggsound \
  > /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/vggsound/download_vggsound_hfmirror.log 2>&1 < /dev/null &
```

目标结构：

```bash
raw/vggsound/scratch/
```

脚本扫描：

```bash
raw/vggsound/scratch/
```

### 5.7 Daily-Omni

下载源：

- ModelScope 未找到。
- HF-Mirror 推荐：<https://hf-mirror.com/datasets/liarliar/Daily-Omni>
- HuggingFace 原始地址：<https://huggingface.co/datasets/liarliar/Daily-Omni>
- GitHub：<https://github.com/lliar-liar/daily-omni>

下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/daily_omni

setsid nohup huggingface-cli download \
  --repo-type dataset liarliar/Daily-Omni \
  --local-dir /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/daily_omni \
  > /data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/daily_omni/download_daily_omni_hfmirror.log 2>&1 < /dev/null &
```

目标结构：

```bash
raw/daily_omni/video/
raw/daily_omni/audio/
```

脚本扫描：

```bash
raw/daily_omni/video/
```

## 6. raw 目录整理要求

下载完成后，必须整理成脚本期望的 `raw/` 结构：

```bash
/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw/
  daily_omni/video/*.mp4
  worldsense/videos/*.mp4
  hdtf/videos/*.mp4
  avatar/**/*.mp4
  vggsound/scratch/**/*.mp4
  vgg_monoaudio/inter_class/mixed/**/*.mp4
  voxceleb/vox2_mp4/dev/**/*.mp4
```

脚本固定扫描映射：

```bash
daily_omni=video
worldsense=videos
hdtf=videos
avatar=.,video
vggsound=scratch
vgg_monoaudio=inter_class/mixed
voxceleb=vox2_mp4/dev
```

整理完成后检查：

```bash
DATA_ROOT=/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval

test -d "$DATA_ROOT/raw/daily_omni/video"
test -d "$DATA_ROOT/raw/worldsense/videos"
test -d "$DATA_ROOT/raw/hdtf/videos"
test -d "$DATA_ROOT/raw/avatar"
test -d "$DATA_ROOT/raw/vggsound/scratch"
test -d "$DATA_ROOT/raw/vgg_monoaudio/inter_class/mixed"
test -d "$DATA_ROOT/raw/voxceleb/vox2_mp4/dev"

find "$DATA_ROOT/raw/daily_omni/video" -name "*.mp4" | sed -n "1,3p"
find "$DATA_ROOT/raw/worldsense/videos" -name "*.mp4" | sed -n "1,3p"
find "$DATA_ROOT/raw/hdtf/videos" -name "*.mp4" | sed -n "1,3p"
find "$DATA_ROOT/raw/avatar" -name "*.mp4" | sed -n "1,3p"
find "$DATA_ROOT/raw/vggsound/scratch" -name "*.mp4" | sed -n "1,3p"
find "$DATA_ROOT/raw/vgg_monoaudio/inter_class/mixed" -name "*.mp4" | sed -n "1,3p"
find "$DATA_ROOT/raw/voxceleb/vox2_mp4/dev" -name "*.mp4" | sed -n "1,3p"
```

如果任何路径缺失，停止。不要现场改代码绕过路径。

## 7. 启动 Qwen3-Omni 服务

推荐 4 张 GPU：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup python -m vllm.entrypoints.openai.api_server \
  --model /data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct \
  --served-model-name qwen3-omni-30b-a3b-instruct \
  --host 127.0.0.1 \
  --port 8093 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.86 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --dtype bfloat16 \
  --enforce-eager \
  > logs/qwen3_omni_8093_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

echo $! | tee logs/qwen3_omni_8093.pid
```

健康检查：

```bash
curl -fsS http://127.0.0.1:8093/v1/models
curl -fsS http://127.0.0.1:8093/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-omni-30b-a3b-instruct","messages":[{"role":"user","content":"Reply with OK only."}],"max_tokens":8}'
```

如果 `/v1/models` 能回但 `/v1/chat/completions` 长时间无响应，说明 vLLM 假死，需要重启服务后再跑构造。

## 8. 先跑 0.1% A+B smoke

新服务器先不要直接全量跑。建议先按每个数据集 0.1% 做 A+B smoke，确认：

- 7 个数据集都能产生 clip folders。
- `single_source_annotations.jsonl` 持续增长。
- `a_visual_audio_anchor_triplets.jsonl` 存在。
- `b_speech_audio_content_triplets.jsonl` 存在。
- `b_all_audio_cvr_triplets.jsonl`、`b_main_audio_cvr_triplets.jsonl`、`b_extended_audio_cvr_triplets.jsonl`、`b_diagnostic_asr_risk_triplets.jsonl` 生成。
- `b_splits/split_summary.json` 生成。
- `manual_review/A/` 和 `manual_review/B/` 都有样本。

smoke 失败时，只回传日志和 summary，不要现场 patch 代码。

## 9. A+B 全量运行命令

smoke 通过后，全量跑 A-line + B-line：

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
git pull --ff-only origin main

source /data02/usr/wangqihao/miniconda3/etc/profile.d/conda.sh
conda activate omni_src

mkdir -p logs

RUN_ROOT=/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs/audio_cvr_ab_6_9s_full_$(date +%Y%m%d_%H%M%S)
LOG=logs/audio_cvr_ab_6_9s_full_$(date +%Y%m%d_%H%M%S).log

setsid nohup bash scripts/run_audio_cvr_ab_6_9s_full_4gpu.sh \
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
  --target-a-count 1000000 \
  --target-b-count 1000000 \
  > "$LOG" 2>&1 < /dev/null &

echo $! | tee logs/audio_cvr_ab_6_9s_full.pid
echo "$RUN_ROOT"
echo "$LOG"
```

说明：

- 该脚本同时跑 A-line 和 B-line。
- A-line 输出 `a_visual_audio_anchor_triplets.jsonl`。
- B-line 输出 `b_speech_audio_content_triplets.jsonl` 及 tier 分层文件。
- 脚本会自动构建 `clips/audio_cvr_6_9s/`。
- 脚本不会跑 e5、AVIGATE、agent。
- 脚本不会修改原始 `raw/` 数据。
- `target-a-count 1000000` 和 `target-b-count 1000000` 表示尽量保留全部 accepted，后续再筛选。

## 10. 缓存与断点续跑

不要删除 run 目录，不要删除 clip 目录。

缓存机制：

- 切片：每个 mp4 完成后写入 `clips/audio_cvr_6_9s/`，重跑可复用。
- annotation：每条写入 `single_source_annotations.jsonl`，中断后按 `clip_id` 复用。
- audio refresh：每条写入 refresh annotations，重跑可复用。
- propose：每条写入 `accepted_progress_*.jsonl` / `rejected_progress_*.jsonl`。
- merge：可从 ranked/progress JSONL 重新生成 summary 和 review bundle。

如果 vLLM 假死：

1. 停掉当前构造脚本。
2. 重启 8093 Qwen3-Omni 服务。
3. 使用同一个 `RUN_ROOT` 重跑，不要换目录。
4. 不要删除已有 JSONL cache。

## 11. 进度检查

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main

PID=$(cat logs/audio_cvr_ab_6_9s_full.pid)
ps -p "$PID" -o pid,pgid,stat,etime,cmd || true

LOG=$(ls -t logs/audio_cvr_ab_6_9s_full_*.log | head -1)
tail -100 "$LOG"

RUN_ROOT=$(ls -td runs/audio_cvr_ab_6_9s_full_* | head -1)
echo "$RUN_ROOT"

wc -l "$RUN_ROOT/single_source_annotations.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/a_candidates.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/b_candidates.jsonl" 2>/dev/null || true
cat "$RUN_ROOT"/a_shards/accepted_progress_*.jsonl 2>/dev/null | wc -l
cat "$RUN_ROOT"/a_shards/rejected_progress_*.jsonl 2>/dev/null | wc -l
cat "$RUN_ROOT"/b_shards/accepted_progress_*.jsonl 2>/dev/null | wc -l
cat "$RUN_ROOT"/b_shards/rejected_progress_*.jsonl 2>/dev/null | wc -l

nvidia-smi -i 0,1,2,3
```

## 12. 最终验收

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
RUN_ROOT=$(ls -td runs/audio_cvr_ab_6_9s_full_* | head -1)
echo "$RUN_ROOT"

cat "$RUN_ROOT/summary.json"
cat "$RUN_ROOT/b_splits/split_summary.json" 2>/dev/null || true

wc -l "$RUN_ROOT/a_visual_audio_anchor_triplets.jsonl"
wc -l "$RUN_ROOT/b_speech_audio_content_triplets.jsonl"
wc -l "$RUN_ROOT/b_all_audio_cvr_triplets.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/b_main_audio_cvr_triplets.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/b_extended_audio_cvr_triplets.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/b_diagnostic_asr_risk_triplets.jsonl" 2>/dev/null || true

find "$RUN_ROOT/manual_review/A" -maxdepth 2 -type f | head -30
find "$RUN_ROOT/manual_review/B" -maxdepth 2 -type f | head -30
```

验收重点：

- `a_visual_audio_anchor_triplets.jsonl` 存在且有样本。
- `b_speech_audio_content_triplets.jsonl` 存在且有样本。
- `b_all_audio_cvr_triplets.jsonl`、`b_main_audio_cvr_triplets.jsonl`、`b_extended_audio_cvr_triplets.jsonl`、`b_diagnostic_asr_risk_triplets.jsonl` 生成。
- `summary.json` 包含 A/B 计数和 B-line 分层统计。
- `manual_review/A/` 和 `manual_review/B/` 都有可人工审核样本。
- 日志没有持续大量 `Input length exceeds`、`fallback_pair_proposal`、`timeout`。
- B-line 使用的是 `b_audio_blind_review_v2`。

## 13. B-line 分层口径

大规模构造时不要因为 ASR-risk 直接停止或删除样本。当前策略是先保留所有 B accepted，再在 merge 阶段分层：

- `B-main`：低 ASR 风险、高视频语境、强 audio delta，用于论文主 benchmark。
- `B-extended`：中等风险、质量合格，用于训练或预训练。
- `B-diagnostic`：ASR-risk、generic talking-head、transcript-like edit 等样本，只做附录诊断。

VoxCeleb 规则：

- VoxCeleb 默认不直接进入 `B-main`。
- 普通 talking-head / ASR-like 样本进入 `B-extended` 或 `B-diagnostic`。
- 只有 `video_context_strength` 高、`asr_degeneracy_risk` 低、`visual_shortcut_risk=false`、`audio_only_verification.accept=true`、`video_only_shortcut.can_identify_target_without_audio=false` 时，才允许进入 `B-main`。

兼容输出：

- `b_speech_audio_content_triplets.jsonl` 等同所有 B accepted。
- 论文主表优先使用 `b_main_audio_cvr_triplets.jsonl`。
- 训练可使用 `b_extended_audio_cvr_triplets.jsonl` 或 main+extended 组合。

## 14. 禁止事项

服务器执行人员必须遵守：

- 不要改代码。
- 不要修改原始数据。
- 不要覆盖旧切片目录。
- 不要删除当前 `RUN_ROOT`。
- 不要跑 e5。
- 不要跑 AVIGATE。
- 不要跑 agent。
- 不要启动 VACE 或任何视频生成模型。
- 不要把纯 ASR / 纯音频数据混入 `B-main`。
- 如果失败，只回传日志和状态，不现场 patch。

## 15. 失败时回传信息

```bash
cd /data02/usr/wangqihao/Demo/test/cvr_clean_main
git rev-parse --short HEAD
git status --short

LOG=$(ls -t logs/audio_cvr_ab_6_9s_full_*.log | head -1)
tail -200 "$LOG"

RUN_ROOT=$(ls -td runs/audio_cvr_ab_6_9s_full_* | head -1)
echo "$RUN_ROOT"
cat "$RUN_ROOT/summary.json" 2>/dev/null || true
wc -l "$RUN_ROOT/single_source_annotations.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/a_candidates.jsonl" 2>/dev/null || true
wc -l "$RUN_ROOT/b_candidates.jsonl" 2>/dev/null || true
cat "$RUN_ROOT"/a_shards/accepted_progress_*.jsonl 2>/dev/null | wc -l
cat "$RUN_ROOT"/b_shards/accepted_progress_*.jsonl 2>/dev/null | wc -l

curl -sS http://127.0.0.1:8093/v1/models || true
nvidia-smi
```
