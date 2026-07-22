# ImageBind runtime provenance

This directory contains the minimal inference-time model code and BPE vocabulary
from `nielsrogge/ImageBind`, branch `feature/add_hf`, pinned at commit:

```text
5120b6bbed3f175bf004895809b628f1b0bcb72f
```

Upstream: https://github.com/nielsrogge/ImageBind

The model code is derived from Meta's ImageBind release. The upstream license is
preserved in `LICENSE`. The local compatibility changes remove the optional
`iopath` dependency from the tokenizer, use Python's built-in `open` for the
local BPE vocabulary, and avoid importing the optional upstream `data` module
from package initialization.

The project-specific video and audio preprocessing lives in
`app/audio_cvr_external_baseline.py`; it uses PyAV because the target server does
not provide `pytorchvideo` and its current torchaudio build cannot decode MP4
containers directly.
