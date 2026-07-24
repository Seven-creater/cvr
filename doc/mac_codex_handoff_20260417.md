# Mac Codex Handoff

Last updated: 2026-04-17

This document is for onboarding a new Codex instance on a Mac. It is written to be read directly by Codex or by a human operator. It covers:

1. What to copy from the Windows machine
2. What to clone or reinstall
3. What exists locally in the repo
4. What exists on the remote server
5. The current experiment state
6. The exact assumptions behind the project

This file is the primary source of truth for migration. Do not rely only on local Codex history databases.

## 1. Short Answer

Do not try to migrate only "chat history" and expect that to be enough.

Use this order:

1. Clone the repo on the Mac
2. Read this document first
3. Reinstall lightweight local dependencies on the Mac
4. Reconnect to the remote server workflow
5. Optionally copy selected `.codex` state from Windows to preserve skills, memories, and session artifacts

Safest rule:

- Use Git for code
- Use this handoff document for project context
- Use copied `.codex` directories only as a best-effort supplement

## 2. What Is Safe To Migrate From Windows

Windows Codex home currently lives under:

- `C:\Users\29785\.codex`

Important subdirectories and files observed there:

- `skills/`
- `memories/`
- `rules/`
- `sessions/`
- `archived_sessions/`
- `worktrees/`
- `sqlite/`
- `config.toml`
- `auth.json`
- `session_index.jsonl`
- `state_5.sqlite`

Recommended migration policy:

### Copy these

These are worth copying to the Mac:

- `skills/`
- `memories/`
- `rules/`
- this repo itself via Git, not by raw folder copy if possible

These are optional best-effort copies:

- `sessions/`
- `archived_sessions/`
- `session_index.jsonl`
- `sqlite/`
- `state_5.sqlite`

Reason:

- skills, memories, and rules are plain project guidance
- sessions/sqlite files may be useful, but portability across machines and app versions is not guaranteed

### Do not rely on copying these blindly

- `auth.json`
- `config.toml`

Reason:

- `auth.json` may contain machine-specific login state or secrets
- `config.toml` may include Windows-specific paths

Best practice:

- re-login on the Mac instead of depending on copied auth state
- review `config.toml` manually before reusing it

## 3. Suggested Migration Procedure

### 3.1 On the Mac, set up Codex first

Make sure Codex is installed and can open repositories normally.

### 3.2 Clone the repo on the Mac

Use Git, not manual folder copy.

Repo remote:

- `https://github.com/Seven-creater/cvr.git`

Current important branch:

- `main`

Current known HEAD when this document was written:

- `aae5089d87734af0a917aab360e9f4e89a4dc9b0`

Commit message:

- `parallelize omni descriptions and cache videos`

### 3.3 Copy selected `.codex` state

From Windows:

- `C:\Users\29785\.codex\skills`
- `C:\Users\29785\.codex\memories`
- `C:\Users\29785\.codex\rules`

Optionally also copy:

- `C:\Users\29785\.codex\sessions`
- `C:\Users\29785\.codex\archived_sessions`
- `C:\Users\29785\.codex\session_index.jsonl`
- `C:\Users\29785\.codex\sqlite`

On the Mac, copy them into:

- `~/.codex/skills`
- `~/.codex/memories`
- `~/.codex/rules`
- optional matching session paths

If there is any conflict, prefer:

- keeping the Mac install working
- copying only the readable plain-text guidance first

### 3.4 Read this document inside Codex on the Mac

Tell Mac Codex:

- read `doc/mac_codex_handoff_20260417.md`
- treat it as the main project handoff
- do not assume local Mac paths are the same as Windows paths

## 4. Local Repo State

This project is a retrieval + Omni reranking workflow around AVIGATE.

Core purpose:

- use official AVIGATE retrieval as the base ranking
- use Omni (Qwen2.5-Omni via vLLM service) to add extra understanding and rerank top-k results
- evaluate both V2T and T2V

Important recent commits:

- `aae5089` `parallelize omni descriptions and cache videos`
- `af2bf97` `tighten t2v prompts and expose target ids`
- `eebb522` `add partial-eval sharding and merge`
- `d43ce42` `cache AVIGATE corpus encodings`
- `e73d9ec` `add omni-guided official rerank flow`

Current local unit test status at time of writing:

- `Ran 19 tests ... OK`

## 5. Important Code Files

Main files in this repo:

- [avigate_official.py](C:\Users\29785\.codex\worktrees\635f\research\app\avigate_official.py)
- [avigate_agent.py](C:\Users\29785\.codex\worktrees\635f\research\app\avigate_agent.py)
- [omni_checker.py](C:\Users\29785\.codex\worktrees\635f\research\app\omni_checker.py)
- [eval.py](C:\Users\29785\.codex\worktrees\635f\research\app\eval.py)

Tests:

- [test_avigate_official.py](C:\Users\29785\.codex\worktrees\635f\research\tests\test_avigate_official.py)
- [test_avigate_agent.py](C:\Users\29785\.codex\worktrees\635f\research\tests\test_avigate_agent.py)
- [test_omni_checker.py](C:\Users\29785\.codex\worktrees\635f\research\tests\test_omni_checker.py)

## 6. Current Retrieval Architecture

### 6.1 Official baseline

AVIGATE official retrieval is the base system.

This is the "paper-faithful" retrieval part.

Known official baseline numbers from prior server validation:

- T2V: `R@1=0.464`, `R@5=0.732`, `R@10=0.827`
- V2T: `R@1=0.435`, `R@5=0.723`, `R@10=0.819`

These are the values to compare against for `round1` on full runs.

### 6.2 Agent / reranking layer

There are two pipelines:

#### V2T

1. Omni describes the query video
2. Official AVIGATE retrieves top-k texts
3. Omni reranks the candidate texts

#### T2V

1. Omni understands the query text
2. Official AVIGATE retrieves top-k videos
3. Omni describes top candidate videos
4. Omni reranks the candidate videos

### 6.3 Metrics

Current evaluation output distinguishes:

- `round1_recall`
- `final_recall`
- `final_top1_accuracy`

Interpretation:

- `round1_recall` = official AVIGATE retrieval before Omni reranking
- `final_recall` = after Omni reranking

For small-sample experiments (48, 64, 80), use these for within-batch comparisons only.

Do not compare 48-sample absolute values directly against the paper full baseline.

## 7. Efficiency Improvements Already Implemented

### 7.1 AVIGATE corpus cache

Implemented in the repo.

Purpose:

- avoid re-encoding the entire text/video corpus every run

Observed result on server:

- first V2T smoke run: about 540s
- second run using cache: about 60s

Known cache file example:

- `9a5f4f781c44b2e6.pt` (~2.5GB)

### 7.2 Sharded partial/full evaluation

Implemented in the repo.

Purpose:

- split evaluation by `start_index` and `sample_size`
- run multiple shards on GPUs 4,5,6,7
- merge results afterwards

Relevant CLI support:

- `--start-index`
- `avigate-agent-merge`

### 7.3 Omni-side efficiency improvements

Implemented in the repo.

Purpose:

- cache `describe_video(video_id)`
- parallelize candidate video descriptions for T2V

Relevant CLI support:

- `--omni-concurrency`

Recommended initial value:

- `2`

## 8. Remote Server State

The heavy experiments run on a remote Linux server, not locally on Windows or Mac.

### 8.1 Omni service

Known service:

- `http://127.0.0.1:8092/v1`

Known status during recent runs:

- listening on `127.0.0.1:8092`
- backed by Qwen2.5-Omni via vLLM

Important policy:

- do not restart Omni unless necessary
- if restart is required, use GPU memory utilization `0.70`

GPU allocation policy used in experiments:

- GPU 2,3: Omni service
- GPU 4,5,6,7: AVIGATE experiments
- GPU 0,1: intentionally left free in recent runs

### 8.2 AVIGATE-related server paths

Checkpoint:

- `/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/avigate/ckpt_msrvtt_paper_like_4gpu_stable/pytorch_model.bin.4`

Model dir:

- `/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/avigate/ckpt_msrvtt_paper_like_4gpu_stable`

Data JSON:

- `/data02/pretrained_model/cvr_learn/cvr_data/03_general_video_text/msr-vtt/AVIGATE/MSRVTT/MSRVTT_data.json`

Split CSV:

- `/data02/pretrained_model/cvr_learn/cvr_data/03_general_video_text/msr-vtt/AVIGATE/MSRVTT/MSRVTT_JSFUSION_test.csv`

Video root:

- `/data02/pretrained_model/cvr_learn/cvr_data/03_general_video_text/msr-vtt/AVIGATE/MSRVTT/videos/all_compressed`

Audio root:

- `/data02/pretrained_model/cvr_learn/cvr_data/03_general_video_text/msr-vtt/AVIGATE/MSRVTT/videos/audio_all_compressed`

CLIP weight:

- `/data02/pretrained_model/cvr_learn/cvr_model/01_lightweight_task_specific/clip/ViT-B-32.pt`

Cache dir used in recent runs:

- `/data02/usr/wangqihao/Demo/test/cvr/runs/official_rerank_20260416/cache`

## 9. Current Experiment Findings

### 9.1 Small-sample results should be treated as A/B experiments

48-sample runs are useful for direction checking, not for absolute comparison to the paper.

Because:

- `1 / 48 = 2.08%`

So a difference of one sample changes metrics by 2.08 percentage points.

### 9.2 Pre-optimization 48-sample result

Before Omni parallelization/caching improvements, one observed 48-sample result was:

#### V2T

- round1: `R@1=0.3958`, `R@5=0.7083`, `R@10=0.7708`
- final: `R@1=0.4792`, `R@5=0.6875`, `R@10=0.7708`

Interpretation:

- improves top1
- does not reliably improve top5/top10

#### T2V

- round1: `R@1=0.3333`, `R@5=0.6250`, `R@10=0.7917`
- final: `R@1=0.3958`, `R@5=0.6250`, `R@10=0.7917`

Interpretation:

- improves top1 on that batch
- no gain on top5/top10

### 9.3 Diagnostic correction

Two earlier T2V diagnoses were partially wrong:

1. `target_video_id` was not broken; it simply was not written into the trace before the `af2bf97` change.
2. `candidate_video_descriptions` were not actually empty; an analysis script read the wrong field level.

Correct candidate structure:

```json
{
  "rank": ...,
  "candidate": {...},
  "video_description": {...}
}
```

So:

- `video_id` is under `candidate.video_id`
- `summary` is under `video_description.summary`

## 10. What The Mac Codex Should Do First

When Mac Codex starts on this project, it should do this:

1. read this file first
2. inspect the latest repo HEAD
3. do not assume local Mac GPU execution is possible
4. treat the server as the execution environment for heavy runs
5. use the repo code as source of truth, not only copied local session history

Suggested first prompt for Mac Codex:

> Read `doc/mac_codex_handoff_20260417.md`, summarize the current repo architecture, then prepare server-side commands only after confirming the current git HEAD and the available evaluation CLI options.

## 11. What To Install On The Mac

The Mac does not need the full Linux GPU stack if it is only being used for code work and orchestration.

Recommended minimum on the Mac:

- Git
- Python
- the repo itself
- basic Python test dependencies needed to run the local unit tests

Do not try to run the server GPU experiment locally on the Mac unless you intentionally rebuild the entire AVIGATE and Omni runtime for macOS.

The Mac's main roles should be:

- editing code
- reading docs
- generating commands
- reviewing results

The remote Linux server's roles should remain:

- AVIGATE execution
- Omni inference
- full evaluation

## 12. Migration Checklist

Use this checklist on the Mac:

- [ ] Install Codex
- [ ] Clone `https://github.com/Seven-creater/cvr.git`
- [ ] Read this handoff document
- [ ] Copy `skills/`, `memories/`, and `rules/` from Windows `.codex`
- [ ] Re-login to Codex rather than depending on copied `auth.json`
- [ ] Confirm repo HEAD
- [ ] Confirm latest unit tests pass locally if needed
- [ ] Treat the Linux server as the execution environment for heavy runs
- [ ] Reuse existing server cache directories where possible

## 13. Final Notes

The main risk during migration is not code loss. The main risk is losing the operational context:

- which machine owns the GPU workloads
- which paths are server-only
- which results are full-baseline values
- which results are only small-batch A/B experiments

That is why this document should be read before any new experiment is launched from the Mac.
