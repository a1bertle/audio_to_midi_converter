# Implementation Plan: Replace HTDemucs with Mel-Band RoFormer via audio-separator

**Date:** 2026-04-26
**Session:** `research/2026-04-26_vocal-stem-separation-alternatives/`
**Status:** Pre-implementation (awaiting validation gate)

---

## Objective

Replace `htdemucs` (demucs 4.0.1, 4-stem, MPS) in the vocal transcription pipeline
with Mel-Band RoFormer via `audio-separator`, targeting:

| Metric | Target | Baseline |
|--------|--------|----------|
| Vocals/other cross-leakage | ≤ −7.5 dB | −4.65 dB [measured] |
| Vocals SRR | ≥ +12.79 dB | +12.79 dB [measured] |
| RMVPE hallucination ratio | ≤ 8% | 13.3% [measured] |

---

## Phase 0 — Environment Setup

### 0.1 Install audio-separator into existing venv

```bash
source .venv/bin/activate
pip install audio-separator
```

Expected: pulls in onnxruntime (CPU), torch (existing), numpy, soundfile.
Verify: `audio-separator --version`

### 0.2 Identify MBR model filename

```bash
python3 -c "import audio_separator; print(audio_separator.__file__)"
# Check default model list or run:
audio-separator --list_models | grep -i "mel\|mbr\|roformer"
```

Target model: `MelBand-Roformer.ckpt` or `mel_band_roformer_vocals.onnx`.
Fallback: BS-RoFormer equivalent (`bs_roformer_vocals.onnx`).

---

## Phase 1 — Validation Benchmark

Run validation BEFORE any code changes. Gate: must pass both target metrics.

### 1.1 Run audio-separator on benchmark input

```bash
source .venv/bin/activate
audio-separator \
  "inputs/Foals - My Number (Official Audio).mp3" \
  --model_filename "mel_band_roformer_vocals.onnx" \
  --output_dir "research/2026-04-26_vocal-stem-separation-alternatives/stems/" \
  --output_format wav
```

Time the run and record RTF:
```python
# or wrap in time.perf_counter() via validate.sh
```

### 1.2 Evaluate stems with existing evaluator

```bash
python3 tests/evaluation/evaluate_stems.py \
  --mix "inputs/Foals - My Number (Official Audio).mp3" \
  --stems-dir "research/2026-04-26_vocal-stem-separation-alternatives/stems/" \
  --json-out "research/2026-04-26_vocal-stem-separation-alternatives/eval_mbr.json"
```

Record: vocals SRR, vocals/other cross-leakage. Compare to htdemucs_4s baseline.

### 1.3 Decision gate

If vocals/other XL ≤ −7.5 dB AND vocals SRR ≥ +12.79 dB:
→ Proceed to Phase 2.
Else:
→ Try BS-RoFormer model (step 1.1 with `bs_roformer_vocals.onnx`).
→ If still failing: document in notes.md and close session without code changes.

---

## Phase 2 — Code Change

**File:** `audio2midi/transcribers/rmvpe.py`

### 2.1 Replace `_separate_vocals` function

**Current implementation** (`rmvpe.py:134–167`):
- Uses `python3 -m demucs --name htdemucs` subprocess.
- Expects output in `out_dir/htdemucs/<stem>/vocals.wav`.

**New implementation:**
- Use `audio_separator.Separator` Python API (not CLI subprocess).
- Run MBR model for vocal stem.
- Output path: `out_dir/audio_separator/<stem>/Vocals.wav` (capital V — verify actual filename).
- Fallback: if `audio-separator` not installed, fall back to htdemucs with warning.

**Replacement function sketch** (in `rmvpe.py:134`):

```python
def _separate_vocals(audio_path: Path, out_dir: Path) -> Path:
    """Run Mel-Band RoFormer vocal separation and return path to vocals.wav."""
    try:
        from audio_separator.separator import Separator  # noqa: F401
    except ImportError:
        _LOG.warning(
            "audio-separator not installed; falling back to htdemucs. "
            "Install with: pip install audio-separator"
        )
        return _separate_vocals_htdemucs(audio_path, out_dir)

    sep_out = out_dir / "audio_separator"
    sep_out.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        model_file_dir=str(out_dir / "models"),
        output_dir=str(sep_out),
        output_format="wav",
    )
    separator.load_model(model_filename="mel_band_roformer_vocals.onnx")
    output_files = separator.separate(str(audio_path))

    # audio-separator emits "Vocals" and "Instrumental" stems
    vocals_path = _find_vocals_file(sep_out, audio_path.stem)
    if not vocals_path or not vocals_path.exists():
        raise RuntimeError(
            f"audio-separator did not produce a vocals stem. Files: {output_files}"
        )
    LOGGER.info("Vocals stem (MBR): %s", vocals_path)
    return vocals_path
```

Rename old function to `_separate_vocals_htdemucs` as the fallback.

Add `_find_vocals_file` helper to glob for `*Vocals*.wav` or `*vocals*.wav` in stem dir.

### 2.2 Update model name constant

Add module-level constant near line 130:
```python
_MBR_MODEL = "mel_band_roformer_vocals.onnx"
```

### 2.3 Update docstring in `RmvpeTranscriber` (line 462)

Update: `htdemucs → RMVPE` → `MBR (Mel-Band RoFormer) → RMVPE`.

### 2.4 Update `requirements.txt`

Add:
```
audio-separator>=0.24.0
```

Keep `demucs>=4.0.0` as optional fallback dependency (note in comment).

---

## Phase 3 — Validation Post-Implementation

### 3.1 Run unit smoke test

```bash
source .venv/bin/activate
python3 -m audio2midi \
  --input "inputs/Foals - My Number (Official Audio).mp3" \
  --stems-only
# Verify vocals.wav produced in workdir/stems/
```

### 3.2 Run full hallucination eval on Adounravel track (existing output)

Re-run `scripts/eval_vocal_midi.py` on a freshly separated vocals stem from
MBR to compare hallucination ratio before/after.

```bash
python3 scripts/eval_vocal_midi.py \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/Adounravel_歌いました.mid" \
  "<mbr_vocals_path>" \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/extracted/eZuDklxsaRg.wav"
```

This is also run automatically in `validate.sh` Step 4.

---

## Phase 4 — Documentation

- Update `README.md`: replace htdemucs install instruction with audio-separator.
- Update `requirements.txt` and `requirements-notebook.txt` if notebook uses stems.
- Update `audio2midi/transcribers/rmvpe.py` module docstring.

---

## Rollback Plan

If MBR produces worse hallucination despite better SDR (e.g., different bleed
character that confuses RMVPE):
1. Revert `_separate_vocals` to htdemucs via `git revert`.
2. Document failure mode in `notes.md`.
3. Consider post-processing approach: apply a narrowband gate on the MBR vocals
   output targeting the F0 fundamental range (80–1400 Hz) before RMVPE.

---

## Validation Script: `validate.sh`

See `validate.sh` in this session directory.
