# Audio-CVR AAAI-27 Paper Toolchain

## Governing Sources

The paper is governed by the following sources, in descending order of
authority:

1. `doc/aaai_audiocvr_paper_storyline_canonical.md`
2. Frozen experiment outputs and their recorded SHA256 values
3. The official AAAI-27 author kit
4. Primary papers and official benchmark documentation
5. Installed writing, figure, statistics, and review skills

Writing or figure tools may improve structure and presentation, but they must
not change the claims, invent evidence, or silently strengthen conclusions.

## Official Author Kit

Downloaded from:

```text
https://aaai.org/wp-content/uploads/2026/05/AuthorKit27.zip
```

Files copied without modification:

```text
aaai2027.sty
aaai2027.bst
ReproducibilityChecklist.tex
ReproducibilityChecklist.pdf
```

Recorded SHA256 values:

```text
aaai2027.sty                  391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623
aaai2027.bst                  5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c
ReproducibilityChecklist.tex  06a3459158089bf1c64b738986118f1d1566e816da4b710c6397561e33c3d5e6
ReproducibilityChecklist.pdf  7fcc703769036e3566daccd59560aaca9b187f49da79fc9d1e13155b61e7dd9e
```

The anonymous manuscript uses:

```tex
\usepackage[submission]{aaai2027}
\usepackage{natbib}
```

AAAI-27 loads its own text, sans-serif, and monospaced fonts. Do not add the
legacy `times`, `helvet`, or `courier` packages to the manuscript preamble.

The generated `main.tex` is the single submission source. It is produced from
`main.template.tex` by `build_paper_assets.py`, which injects generated tables
without requiring `\input` files.

## Installed Nature Skills

Pinned upstream commit:

```text
91862221b39f7ca16d52ae0e1e9cb6c2bb31a96b
```

Installed globally:

```text
nature-shared
nature-writing
nature-polishing
nature-figure
nature-statistics
nature-reviewer
nature-ref-verifier
nature-reader
nature-proposal-writer
```

The saved `nature-figure` backend is `python`. Automatic skill updates are
disabled so that the workflow cannot change immediately before submission.

## Build

Install the pinned figure dependencies in an isolated environment if needed:

```powershell
python -m venv paper/.venv-figures
paper/.venv-figures/Scripts/python -m pip install -r paper/requirements-figures.txt
```

Generate and export the self-contained Draw.io figures first. The Node exporter
expects Chrome/Chromium and a resolvable Playwright installation:

```powershell
python paper/figures/build_drawio_figures.py
node paper/figures/export_drawio_figures.cjs
```

Then generate tables and the single-source manuscript while preserving those
Draw.io exports:

```powershell
python paper/build_paper_assets.py --strict --skip-figures
```

Strict mode fails if a final Full1000 or ImageBind field is missing. The
`--skip-figures` flag is mandatory for the Draw.io release path; omitting it
invokes the legacy Matplotlib figure functions and overwrites the exported
figure bundle.

Compile with the installed LaTeX plugin or directly with the available TeX
runtime. The expected output is `paper/build_main/main.pdf`.

On this Windows workspace, put TinyTeX before MiKTeX so that `latexmk` has a
working Perl runtime:

```powershell
$env:PYTHONUTF8 = "1"
$env:PATH = "C:\Users\29785\AppData\Roaming\TinyTeX\bin\windows;$env:PATH"
python C:\Users\29785\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py `
  C:\Users\29785\.codex\worktrees\6666\research\paper\main.tex `
  --compiler texlive `
  --output-directory C:\Users\29785\.codex\worktrees\6666\research\paper\build_main `
  --json
```

The local TinyTeX runtime needed the official-template dependencies `newtx`,
`xpatch`, `xstring`, `mweights`, `fontaxes`, `carlisle`, `placeins`, and
`courier`.

## AI-Assistance Boundary

Before submission, re-check the current AAAI-27 policy. Until a more permissive
policy is confirmed, use AI assistance only to organize, edit, and polish
author-provided research material. Authors must verify every sentence,
reference, number, figure, and table and remain responsible for the submission.
