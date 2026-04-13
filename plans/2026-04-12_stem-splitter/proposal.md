# Proposal: Stem Splitter Feature

**Date:** 2026-04-12
**Status:** Draft

---

## Problem Statement

The `audio2midi` pipeline currently ingests a full mixed audio track and
transcribes it directly. Mixed audio degrades transcription quality because
polyphonic content from multiple instruments interferes with per-instrument
pitch detection. A stem splitter would separate the input mix into individual
instrument stems (vocals, drums, bass, piano, guitar, other) before
transcription, enabling the pipeline to transcribe each stem independently
and with higher accuracy.

---

## Success Criteria

- Produce isolated stems as WAV files from a mixed input track.
- Each stem is usable as a drop-in replacement for the current preprocessed
  WAV fed to the transcribers.
- Splitting a typical 3–5 min track at 44.1 kHz runs within a practical
  wall-clock budget on the target machine (M1 MacBook Air, 16 GB unified RAM)
  — target ≤ 3× real-time, i.e. ≤ 15 min for a 5 min track.
- Memory footprint of the splitter stays within a 4 GB headroom (leaving
  headroom for subsequent transcription).
- Quality bar: isolated piano or guitar stem is subjectively cleaner than the
  raw mix (no formal MUSDB18 evaluation required for initial proposal; full
  benchmark left for research phase).

---

## Constraints

- Platform: Apple M1, 7-core GPU, 16 GB unified RAM, macOS (darwin).
- Existing dependencies include `torch>=2.2` — any model must be compatible
  or must have a pure-CPU / Core ML fallback.
- Must not break existing pipeline: stem splitting is an optional pre-step,
  not a forced replacement.
- License: project is GPL-3.0. Selected library must be GPL-compatible.
- No GPU VRAM budget (unified memory only; no discrete CUDA device).

---

## Non-Goals

- Real-time or low-latency stem splitting.
- Formal MUSDB18 SDR benchmark for this proposal (left to research phase).
- Vocals-to-MIDI transcription (only instrumental stems are transcribed).
- Retraining or fine-tuning separation models.

---

## Candidate Approaches

| # | Approach | Summary |
|---|----------|---------|
| 1 | **Demucs (htdemucs)** | Meta's hybrid Transformer+Waveform model; separates 4 stems (drums, bass, other, vocals) or 6 stems (adds piano, guitar); ships with PyTorch weights; supports MPS on Apple Silicon |
| 2 | **Spleeter** | Deezer's TF/Keras model; 2-, 4-, or 5-stem models; older but widely used; CPU-only on Apple Silicon (TF MPS not mature) |
| 3 | **Open-Unmix (UMX)** | Research baseline; PyTorch; 4-stem; lighter weight than Demucs; good MPS compatibility |
| 4 | **NMF / HPSS (signal-processing baseline)** | librosa's harmonic-percussive source separation or NMF; no neural model; fast; low quality for complex mixes |

---

## Research Findings

> No benchmark data exists yet. Values below are marked per provenance policy.

### Approach 1 — Demucs (htdemucs / htdemucs_6s)

- **Model size:** `htdemucs` ~83 MB on disk; `htdemucs_6s` (6-stem, includes
  piano + guitar) ~83 MB [assumed — from public Demucs README; not locally
  measured].
- **MPS support:** Demucs v4 (`demucs>=4.0`) officially supports `mps` device
  on Apple Silicon via PyTorch MPS backend [assumed — from Demucs GitHub
  issues and PyTorch MPS documentation; not benchmarked locally].
- **Stem outputs:** `htdemucs_6s` produces: `drums`, `bass`, `other`,
  `vocals`, `piano`, `guitar` — exactly the stems relevant to this pipeline.
- **PyPI package:** `demucs>=4.0.0` (MIT licence — GPL-3.0 compatible) [assumed
  from PyPI metadata; not verified in a fresh install].
- **Runtime estimate:** Community reports ~1–2× real-time on M1 with MPS [assumed
  — from GitHub issues; no local measurement].

### Approach 2 — Spleeter

- **MPS support:** TensorFlow on Apple Silicon uses the `tensorflow-metal`
  plugin; `tensorflow-metal` support is incomplete and not officially
  recommended for inference workloads as of 2025 [assumed — from TF Metal
  release notes; not tested].
- **Stem granularity:** Max 5 stems; no dedicated piano or guitar stem
  [source-code — from Spleeter model registry].
- **Licence:** MIT [assumed from PyPI metadata].
- **Risk:** TF dependency conflicts with existing PyTorch stack; no piano/guitar
  stem.

### Approach 3 — Open-Unmix (UMX)

- **Model size:** ~136 MB per stem model (4 models for 4-stem) [assumed — from
  Open-Unmix paper appendix; not locally measured].
- **Stems:** 4-stem only (drums, bass, other, vocals); no dedicated piano or
  guitar [source-code — from Open-Unmix repo].
- **PyTorch:** fully compatible [assumed].
- **Limitation:** No piano/guitar stem; `other` residual would contain both,
  limiting downstream transcription quality.

### Approach 4 — HPSS (librosa baseline)

- **Already a dependency:** `librosa>=0.10.2` is in `pyproject.toml`
  [source-code — `pyproject.toml:23`].
- **Separation quality:** HPSS only separates harmonic vs. percussive content;
  cannot isolate piano from guitar or vocals [source-code — librosa HPSS docs].
- **Speed:** Near-instant for typical track lengths [assumed].
- **Use case:** Useful only as a noise-reduction pre-step, not a real stem
  splitter.

---

## Results Summary

| Approach | Piano/Guitar stem? | MPS on M1? | Licence | Dep conflict risk | Quality (subjective) |
|----------|--------------------|------------|---------|-------------------|----------------------|
| Demucs htdemucs_6s | Yes (dedicated) | Yes [assumed] | MIT | Low | High [assumed] |
| Spleeter 5-stem | No (lumped in "other") | No (TF) | MIT | High (TF vs PyTorch) | Medium [assumed] |
| Open-Unmix 4-stem | No ("other" only) | Yes [assumed] | MIT | Low | Medium [assumed] |
| HPSS (librosa) | No (harmonic/percussive only) | N/A | LGPL | None | Low [source-code] |

> All quality and runtime estimates are `[assumed]` pending a research session
> benchmark on real audio. See **Risks** below.

---

## Recommendation

**Approach 1 — Demucs (`htdemucs_6s`)**

Demucs is the only candidate that produces dedicated `piano` and `guitar`
stems, which map directly onto the instruments already supported by the
transcription pipeline. It is PyTorch-native (no new framework dependency),
ships as a PyPI package with an MIT licence (GPL-3.0 compatible), and has
reported MPS support for Apple Silicon. The `htdemucs_6s` 6-stem variant
produces exactly the stems this pipeline needs. All other candidates either
lack piano/guitar stems (UMX, HPSS) or introduce a conflicting TF dependency
with no MPS path (Spleeter).

---

## Tradeoff Analysis

- **Demucs vs. UMX:** UMX is lighter but its `other` stem bundles piano,
  guitar, and everything else — downstream transcription would face the same
  polyphony problem as the raw mix. Demucs `htdemucs_6s` is the only
  off-the-shelf model with dedicated piano and guitar outputs.
- **Demucs vs. Spleeter:** Spleeter introduces TensorFlow, which conflicts with
  the existing PyTorch stack, has no practical MPS path on M1, and provides no
  piano/guitar stem. No upside over Demucs.
- **Demucs vs. HPSS:** HPSS is fast and zero-dependency but only splits
  harmonic from percussive energy — it cannot isolate a piano from a guitar.
  Retain as a cheap preprocessing option (it already works via librosa), but it
  does not satisfy the stem-splitter goal.
- **Runtime risk:** The 3× real-time budget is `[assumed]` to be achievable with
  MPS on M1. If MPS is unavailable or slow, CPU inference may exceed the budget
  for long tracks. This must be measured in the research phase.

---

## Risks and Open Questions

- **[assumed] MPS runtime on M1:** No local benchmark. If htdemucs_6s exceeds
  the 3× real-time budget on CPU fallback, chunked processing or a lighter
  model (`htdemucs` 4-stem + manual isolation) may be needed.
- **[assumed] Memory footprint:** htdemucs_6s peak RAM during inference is
  unknown locally. Must be measured against the 4 GB budget.
- **[assumed] Stem quality on the project's actual audio inputs:** SDR for
  piano/guitar separation in complex band mixes is unknown. The research phase
  should measure subjective quality on representative tracks.
- **Demucs version pinning:** `demucs` is not currently a project dependency;
  version compatibility with `torch>=2.2` must be verified.
- **CLI surface:** The stem splitter should be an optional flag (`--split-stems`,
  `--stem <name>`) rather than a forced step, to preserve backward compatibility.

---

## Visualizations

None produced at proposal stage. The research phase may warrant a notebook
showing stem waveforms and spectrograms before/after separation.

---

## Next Step

Invoke **researcher** with: "Benchmark Demucs htdemucs_6s on M1 MPS and CPU:
measure wall-clock time, peak RAM, and subjective stem quality (piano + guitar)
on one representative track. Record in `research/2026-04-12_demucs-benchmark/`."

Once research confirms the runtime and memory budget, invoke **planner** with:
"Implement optional Demucs htdemucs_6s stem splitting as a pre-step in the
audio2midi pipeline, exposing `--split-stems` and `--stem <name>` CLI flags."
