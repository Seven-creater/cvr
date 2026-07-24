# composed_omni_retrieval 数据目录结构详解

> 根路径: `/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval`
> 总大小: ~625 GB（raw_datasets ~507 GB + raw ~482 GB + 其他）
> 更新日期: 2026-05-14

---

## 目录总览

```
composed_omni_retrieval/
├── raw_datasets/          # 原始数据集压缩包（zip + tar.gz + parquet + split archive）
├── raw/                   # 解压后的原始视频/音频文件
├── clips/                 # 切片后的视频片段（核心可用数据）
├── metadata/              # 元数据索引（JSONL 格式）
├── reports/               # 数据准备过程的报告
├── runs -> /data02/usr/wangqihao/Demo/test/cvr_clean_main/runs  # 软链接
├── pilot_10/              # 初始 pilot 报告（空）
├── caches/                # 缓存（空）
├── captions/              # 字幕/描述（空）
├── pairs/                 # 视频对（空）
└── splits/                # 数据集划分（空）
```

---

## 1. raw_datasets/ — 原始数据集（压缩包）

路径: `/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw_datasets/`

总大小: ~507 GB。

### 1.1 daily_omni（Daily-Omni 数据集）

| 项目 | 详情 |
|---|---|
| **来源** | HuggingFace 数据集，包含视频+音频+QA问答对 |
| **数据量** | test split: 1196 条 |
| **Schema** | `video_id`, `video`(二进制), `audio`(16kHz), `question`, `candidates`(列表), `answer` |
| **用途** | 通用视频理解评测 |
| **总大小** | ~4.9 GB |

### 1.2 hdtf（HDTF 数据集）

| 项目 | 详情 |
|---|---|
| **来源** | HuggingFace `global-optima-research/HDTF` |
| **用途** | B 线 speech_content：同人同场景说话视频 |
| **总大小** | 9.7 GB（videos.zip 4.9 GB + clips.zip 4.8 GB） |

### 1.3 avatar（AVATAR 数据集）

| 项目 | 详情 |
|---|---|
| **来源** | HuggingFace `mipal/AVATAR` |
| **用途** | A 线 audio_event/music 补充 |
| **总大小** | 5.1 GB（video.zip 3.6 GB + metadata.zip 1.6 GB） |

### 1.4 vggsound_seed（VGGSound 数据集种子）

| 项目 | 详情 |
|---|---|
| **来源** | HuggingFace `Loie/VGGSound` |
| **用途** | A 线 sound/music/audio_event |
| **总大小** | 32 GB（vggsound_00.tar.gz 16 GB + vggsound_01.tar.gz 16 GB） |

### 1.5 vgg_monoaudio（VGG-MonoAudio 数据集）

| 项目 | 详情 |
|---|---|
| **来源** | HuggingFace `jnwnlee/vgg-monoaudio` |
| **用途** | A 线补充，音频-视觉配对 |
| **总大小** | 2.9 GB |
| **文件数** | 6,586 个（1,071 mp4 + 1,120 wav + 2 csv） |

无需解压，已直接复制到 `raw/vgg_monoaudio/`。

### 1.6 VoxCeleb（VoxCeleb1+2 完整数据集）

| 项目 | 详情 |
|---|---|
| **来源** | ModelScope `juliuscn/voxceleb`（HF 下载失败后改用 ModelScope） |
| **用途** | B 线同主体 speech pair |
| **总大小** | 354 GB |

目录结构：
```
raw_datasets/VoxCeleb/
├── vox1/
│   ├── vox1_dev_wav_partaa ~ partad   # 音频 split archive（~30 GB）
│   ├── vox1_dev_txt.zip               # 0.1 GB
│   ├── vox1_test_txt.zip
│   ├── vox1_test_wav.zip              # 1.0 GB
│   └── vox1_meta.csv
├── vox2/
│   ├── vox2_dev_mp4_partaa ~ partai   # 视频 split archive（8×30 GB + 8.6 GB = 248.6 GB）
│   ├── vox2_dev_aac_partaa ~ partah   # 音频 split archive（7×10 GB + 2.2 GB = 72.2 GB）
│   ├── vox2_dev_txt.zip               # 1.5 GB
│   └── vox2_meta.csv
```

注意：`partaa~partai` 是 split archive，需 `cat part* > file.zip` 合并后 unzip 解压。正在解压到 `raw/voxceleb/`。

### 1.7 voxceleb_seed（已废弃）

HuggingFace 下载因 allow_pattern 路径错误失败（只下了 36KB），已被 ModelScope 版本替代（见 1.6）。

### 1.8 worldsense（WorldSense 数据集）

| 项目 | 详情 |
|---|---|
| **来源** | HuggingFace 数据集，视频路径引用而非内嵌二进制 |
| **数据量** | test split: 3172 条 |
| **用途** | 世界知识视频理解评测 |
| **总大小** | ~103 GB（zip 分片） |

---

## 2. raw/ — 解压后的原始视频/音频文件

路径: `/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/raw/`

总大小: ~482 GB

### 2.1 raw/daily_omni/ — Daily-Omni

| 子目录 | 文件数 | 大小 | 格式 | 时长 | 分辨率 |
|---|---|---|---|---|---|
| `video/` | 1,196 mp4 | 3.3 GB | H.264 | ~30s | 640×360 |
| `audio/` | 1,196 wav | 1.6 GB | 16kHz PCM | ~30s | — |

命名规则: `test-{parquet}_{行号}_video.mp4` / `_audio.wav`

### 2.2 raw/hdtf/ — HDTF

| 子目录 | 文件数 | 大小 | 格式 | 时长 | 分辨率 |
|---|---|---|---|---|---|
| `videos/` | 400 mp4 | 5.0 GB | H.264 | 30~140s | 不定（390×390, 534×534, 598×598） |
| `clips/` | 16,914 mp4 | 4.9 GB | H.264 | ~3.2s | 同上 |

- `videos/`: 原始长视频，命名如 `RD_Radio10_000.mp4`
- `clips/`: 切片，命名如 `RD_Radio10_000_0_80.mp4`（起止帧号）

### 2.3 raw/avatar/ — AVATAR

| 内容 | 数量 | 说明 |
|---|---|---|
| **子目录** | 5,002 个 | 每个视频一个目录，如 `004KfU7bgyg_00069/` |
| **mp4 视频** | 10,000 个 | ~10s，640×360 |
| **jpg 帧** | 24,266 个 | 抽取的关键帧 |
| **json 标注** | 24,266 个 | 对应帧的元数据 |
| **总大小** | 5.5 GB | |

### 2.4 raw/vggsound/ — VGGSound

| 项目 | 详情 |
|---|---|
| **mp4 文件数** | 20,000 个 |
| **总大小** | 32 GB |
| **时长** | ~10~15s |
| **分辨率** | 1280×720 |
| **目录结构** | `scratch/` 下按视频 ID 分子目录 |

### 2.5 raw/vgg_monoaudio/ — VGG-MonoAudio

| 项目 | 详情 |
|---|---|
| **mp4 数量** | 1,071 个 |
| **wav 数量** | 1,120 个 |
| **csv 数量** | 2 个（intra_class + inter_class metadata） |
| **总大小** | 2.9 GB |
| **时长** | ~8s（mp4 和 wav 一致） |
| **分辨率** | 1280×720（mp4） |

子目录结构：
- `intra_class/target_audio/` — 类内目标音频（wav）
- `intra_class/mixed/` — 类内混合视频（mp4）
- `inter_class/target_audio/` — 类间目标音频（wav）
- `inter_class/mixed/` — 类间混合视频（mp4）

### 2.6 raw/worldsense/ — WorldSense

| 子目录 | 文件数 | 大小 | 说明 |
|---|---|---|---|
| `videos/` | 1,662 mp4 | 18 GB | 640×360, 30~540s |
| `audios/` | 1,662 | 39 GB | 对应音频 |
| `subtitles/` | 1,662 | 16 MB | 对应字幕 |
| **合计** | 4,986 | 56 GB | |

### 2.7 raw/voxceleb/ — VoxCeleb

| 项目 | 详情 |
|---|---|
| **来源** | ModelScope `juliuscn/voxceleb`，从 `raw_datasets/VoxCeleb/` 合并解压 |
| **总大小** | 371 GB |

| 子目录 | 大小 | 文件数 | 内容 |
|---|---|---|---|
| `vox1/` | 40 GB | 307,033 | 153,516 wav + 153,516 txt + meta.csv（1,251 行） |
| `vox2_mp4/` | 255 GB | 1,092,010 | 1,092,009 mp4 + meta.csv（6,113 行） |
| `vox2_aac/` | 76 GB | 1,092,009 | 1,092,009 aac/m4a 音频 |

- **vox1/**: `wav/` 和 `txt/` 子目录，每条音频对应一个 txt 标注
- **vox2_mp4/**: `dev/` 子目录，mp4 视频，~4~9s，224×224
- **vox2_aac/**: `dev/` 子目录，与 mp4 一一对应的纯音频文件
- **vox2 meta**: `vox2_meta.csv` 包含 6,113 个说话人，1,092,009 条视频

---

## 3. clips/ — 切片后的视频片段

路径: `/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/clips/`

总大小: ~3.5 GB。

### 3.1 clips/omni_stable/ — 稳定切片集

| 项目 | 详情 |
|---|---|
| **文件数** | 2,902 mp4 |
| **总大小** | 2.8 GB |
| **时长** | ~4s |
| **分辨率** | 640×360 |
| **来源** | daily_omni 1,191 + worldsense 1,711 |

### 3.2 clips/detective/ — 检测切片集

| 子目录 | 文件数 | 大小 | 时长 |
|---|---|---|---|
| `daily_omni/` | 836 | 394 MB | ~12s |
| `worldsense/` | 648 | 395 MB | ~12s |
| **合计** | 1,484 | 788 MB | |

### 3.3 clips/synthetic/ — 合成视频

1 个文件，3.6 MB，9.1s，640×352，用于测试。

---

## 4. metadata/ — 元数据索引

路径: `/data02/pretrained_model/cvr_learn/cvr_data/composed_omni_retrieval/metadata/`

| 文件 | 行数 | 大小 | 说明 |
|---|---|---|---|
| `raw_assets.jsonl` | 2,858 | 1.1 MB | 原始视频资产索引 |
| `source_clips_all.jsonl` | 2,858 | 4.7 MB | 所有切片清单 |
| `source_clips_pilot50.jsonl` | 50 | 79 KB | Pilot 50 切片子集 |
| `source_rows.jsonl` | 4,368 | 8.9 MB | 原始 parquet 行索引 |

---

## 5. reports/ — 数据准备报告

| 文件 | 说明 |
|---|---|
| `raw_assets_summary.md` | daily_omni 1,196 + worldsense 1,662 = 2,858 个视频 |
| `source_dataset_prepare_summary.md` | 4,368 行 → 2,858 唯一切片 → 50 pilot |

---

## 6. runs/ — 训练运行目录（软链接）

指向: `/data02/usr/wangqihao/Demo/test/cvr_clean_main/runs`（~707 MB）

---

## 7. 其他目录（目前为空）

| 路径 | 说明 |
|---|---|
| `caches/` | 缓存 |
| `captions/` | 字幕/描述 |
| `pairs/` | 视频对 |
| `splits/` | 数据集划分 |
| `pilot_10/reports/` | Pilot 10 报告 |

---

## 数据总览表

| 数据集 | 路径 | 文件数 | 大小 | 时长 | 分辨率 | Omni-CVR 线路 |
|---|---|---|---|---|---|---|
| daily_omni | `raw/daily_omni/` | 2,392 | 4.9 GB | ~30s | 640×360 | 评测 |
| worldsense | `raw/worldsense/` | 4,986 | 56 GB | 30~540s | 640×360 | 评测 |
| HDTF | `raw/hdtf/` | 17,314 | 9.8 GB | 30~140s / clips 3.2s | 不定 | **B 线** |
| VoxCeleb | `raw/voxceleb/` | 2,491,052 | 371 GB | ~4~9s | 224×224 | **B 线** |
| VGGSound | `raw/vggsound/` | 20,000 | 32 GB | ~10~15s | 1280×720 | **A 线** |
| AVATAR | `raw/avatar/` | 58,532 | 5.5 GB | ~10s | 640×360 | **A 线** |
| VGG-MonoAudio | `raw/vgg_monoaudio/` | 6,586 | 2.9 GB | ~8s | 1280×720 | **A 线** |
| omni_stable 切片 | `clips/omni_stable/` | 2,902 | 2.8 GB | ~4s | 640×360 | 训练/评测 |
| detective 切片 | `clips/detective/` | 1,484 | 788 MB | ~12s | 640×360 | 检测 |

### 数据源与 Omni-CVR 线路对应

| 线路 | 数据源 | 用途 |
|---|---|---|
| **B 线** (speech_content) | HDTF, VoxCeleb | 同人同场景说话视频 |
| **A 线** (audio_event/music) | VGGSound, AVATAR, VGG-MonoAudio | 音频内容变化驱动检索 |
| **评测** | daily_omni, worldsense | 通用视频理解评测 |

---

## 数据流关系

```
raw_datasets/                    (原始压缩包: zip + tar.gz + parquet + split archive)
    │
    ├── daily_omni (parquet 内嵌视频二进制)
    │       ↓ 提取 → raw/daily_omni/ (1,196 mp4 + 1,196 wav, 4.9 GB)
    │
    ├── hdtf (videos.zip + clips.zip)
    │       ↓ unzip → raw/hdtf/ (400 videos + 16,914 clips, 9.8 GB)
    │
    ├── avatar (video.zip + metadata.zip)
    │       ↓ unzip → raw/avatar/ (10,000 mp4 + 24,266 jpg + 24,266 json, 5.5 GB)
    │
    ├── vggsound_seed (vggsound_00/01.tar.gz)
    │       ↓ tar xzf → raw/vggsound/ (20,000 mp4, 32 GB)
    │
    ├── vgg_monoaudio (mp4 + wav + csv)
    │       ↓ cp → raw/vgg_monoaudio/ (1,071 mp4 + 1,120 wav, 2.9 GB)
    │
    ├── worldsense (zip 分片)
    │       ↓ unzip → raw/worldsense/ (1,662×3 files, 56 GB)
    │
    └── VoxCeleb (ModelScope split archive)
            ↓ cat+unzip → raw/voxceleb/ (vox1: 307K files, vox2: 2.2M files, 371 GB)

raw/                             (解压后原始数据)
    ↓ 切片处理
clips/
    ├── omni_stable/             (2,902 clips, ~4s)
    ├── detective/               (1,484 clips, ~12s)
    └── synthetic/               (1 test clip)

metadata/                        (JSONL 索引)
```
