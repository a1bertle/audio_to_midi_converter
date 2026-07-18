#!/usr/bin/env python3
"""
eval_vocal_midi.py — Vocal MIDI transcription quality evaluator.

Measures how accurately a MIDI file transcribes the pitch and timing of a
vocal performance by comparing the MIDI note roll against the F0 contour
extracted from the vocals audio stem via pYIN.

Usage
-----
From a pipeline workdir (paths auto-discovered):

    python scripts/eval_vocal_midi.py --workdir outputs/<timestamp>_<name>/

Or with explicit paths:

    python scripts/eval_vocal_midi.py <midi_file> <vocals_wav> <mix_wav>

Arguments
---------
    --workdir     Path to an audio2midi output workdir.  The script discovers:
                    - MIDI:   <workdir>/*.mid  (first match)
                    - vocals: <workdir>/stems/htdemucs/<track>/vocals.wav
                    - mix:    <workdir>/extracted/<track>.wav
    midi_file     (explicit mode) Path to the MIDI file to evaluate.
    vocals_wav    (explicit mode) Path to the isolated vocals stem (e.g. from Demucs).
    mix_wav       (explicit mode) Path to the original audio mix.

Output
------
JSON to stdout with the following top-level keys:

    midi_properties     — note count, pitch range, duration, tempo, velocity stats
    vocals_audio        — sample rate, duration, peak dBFS
    mix_audio           — sample rate, duration, peak dBFS
    vocal_activity      — RMS-based voiced frame count and ratio
    pyin_f0             — pYIN F0 frame count, voiced ratio, pitch stats (MIDI + Hz)
    pitch_coverage      — frame-level coverage, hallucination ratio, pitch error stats
    pitch_histogram     — histogram intersection between pYIN F0 and MIDI note distributions
    onset_regularity    — inter-onset interval (IOI) mean, std, CV
    duration_mismatch_s — |MIDI end − final pYIN-voiced frame| in seconds
    audio_duration_mismatch_s — |MIDI end − vocals WAV duration| in seconds

Progress and errors are written to stderr and to .logs/error.log at the
project root.

Metrics explained
-----------------
coverage_ratio
    Fraction of pYIN-voiced frames that have an active MIDI note.
    Low coverage (< 0.90) means the transcription misses sung passages.

hallucination_ratio
    Fraction of MIDI-active frames where pYIN marks the audio as unvoiced.
    High hallucination (> 0.05) indicates notes fired on silence or bleed.

pitch_error_*_semitones
    |pYIN F0 (MIDI pitch) − MIDI note pitch| on co-active frames.
    Errors > 0.5 st suggest quantization or tracking issues.

ioi_cv
    Coefficient of variation of inter-onset intervals.
    High CV (> 1.0) indicates irregular, fragmented note timing.

pitch_histogram_intersection
    Normalized histogram intersection between pYIN F0 distribution and MIDI
    note pitch distribution.  1.0 = perfect overlap, 0.0 = no overlap.

Dependencies
------------
Requires the project venv with: librosa, pretty_midi, mido, numpy, scipy.
All dependencies are covered by requirements.txt.

Examples
--------
Evaluate using a workdir (recommended):

    source .venv/bin/activate
    python scripts/eval_vocal_midi.py \\
        --workdir "outputs/2026-04-26_12-34-26_Adounravel/"

Evaluate with explicit paths:

    python scripts/eval_vocal_midi.py \\
        "outputs/2026-04-26_12-34-26_Adounravel/Adounravel.mid" \\
        "outputs/2026-04-26_12-34-26_Adounravel/stems/htdemucs/track/vocals.wav" \\
        "outputs/2026-04-26_12-34-26_Adounravel/extracted/track.wav"

Save output for use in an assessment report:

    python scripts/eval_vocal_midi.py --workdir outputs/<workdir>/ \\
        > research/2026-04-26_my-eval/measurements.json
"""
import argparse
import sys
import json
import logging
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio2midi.logging_config import configure_logging

import numpy as np
import librosa
import pretty_midi
import mido

configure_logging(level="INFO", context=f"argv={sys.argv[1:]}")
log = logging.getLogger("eval_vocal_midi")


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------

def midi_properties(midi_path: Path) -> dict:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    mf = mido.MidiFile(str(midi_path))
    all_notes = [n for inst in pm.instruments for n in inst.notes]
    pitches = [n.pitch for n in all_notes]
    durations = [n.end - n.start for n in all_notes]
    velocities = [n.velocity for n in all_notes]
    tempos = pm.get_tempo_changes()[1].tolist()
    return {
        "duration_s": pm.get_end_time(),
        "num_instruments": len(pm.instruments),
        "total_notes": len(all_notes),
        "pitch_min": int(min(pitches)) if pitches else None,
        "pitch_max": int(max(pitches)) if pitches else None,
        "pitch_range_semitones": int(max(pitches) - min(pitches)) if pitches else 0,
        "pitch_mean": float(np.mean(pitches)) if pitches else None,
        "pitch_median": float(np.median(pitches)) if pitches else None,
        "note_duration_mean_s": float(np.mean(durations)) if durations else None,
        "note_duration_median_s": float(np.median(durations)) if durations else None,
        "note_duration_std_s": float(np.std(durations)) if durations else None,
        "note_duration_min_s": float(min(durations)) if durations else None,
        "note_duration_max_s": float(max(durations)) if durations else None,
        "velocity_mean": float(np.mean(velocities)) if velocities else None,
        "velocity_std": float(np.std(velocities)) if velocities else None,
        "tempos_bpm": tempos,
        "ticks_per_beat": mf.ticks_per_beat,
    }


def vocal_activity(audio: np.ndarray, sr: int, frame_len: int = 2048,
                   hop: int = 512, rms_thresh_db: float = -40.0) -> dict:
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    voiced_mask = rms_db > rms_thresh_db
    total = len(rms_db)
    voiced = int(np.sum(voiced_mask))
    return {
        "rms_thresh_db": rms_thresh_db,
        "total_frames": total,
        "voiced_frames": voiced,
        "voiced_ratio": float(voiced / total) if total else 0.0,
        "rms_db_mean": float(np.mean(rms_db)),
        "rms_db_peak": float(np.max(rms_db)),
    }


def pyin_f0(audio: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    """Return F0 in Hz per frame; NaN where pYIN marks unvoiced."""
    log.info("Running pYIN F0 extraction ...")
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop,
    )
    return np.where(voiced_flag, f0, np.nan)


def f0_to_midi_pitch(f0_hz: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.isfinite(f0_hz) & (f0_hz > 0),
            69 + 12 * np.log2(f0_hz / 440.0),
            np.nan,
        )


def midi_pitch_roll(
    pm: pretty_midi.PrettyMIDI,
    sr: int,
    hop: int = 512,
    n_frames: int | None = None,
) -> np.ndarray:
    """Per-frame MIDI pitch aligned to pYIN frame grid; NaN where no note is active."""
    midi_frames = int(math.ceil(pm.get_end_time() * sr / hop)) + 1
    roll = np.full(max(n_frames or 0, midi_frames), np.nan)
    for inst in pm.instruments:
        for note in inst.notes:
            roll[int(round(note.start * sr / hop)):int(round(note.end * sr / hop))] = note.pitch
    return roll


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive frame ranges for contiguous true values."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def hallucination_region_metrics(
    f0_midi: np.ndarray,
    midi_roll: np.ndarray,
    audio: np.ndarray,
    sr: int,
    hop: int = 512,
    rms_thresh_db: float = -40.0,
    adjacency_s: float = 0.1,
) -> dict:
    """Classify MIDI-active/pYIN-unvoiced regions using energy and adjacency."""
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    n = min(len(f0_midi), len(midi_roll), len(rms_db))
    voiced = np.isfinite(f0_midi[:n])
    hallucinated = np.isfinite(midi_roll[:n]) & ~voiced
    adjacency_frames = max(1, int(math.ceil(adjacency_s * sr / hop)))
    regions = []
    class_frames = {
        "onset_or_offset_overhang": 0,
        "low_energy": 0,
        "unvoiced_high_energy": 0,
    }

    for start, end in _contiguous_regions(hallucinated):
        nearby_start = max(0, start - adjacency_frames)
        nearby_end = min(n, end + adjacency_frames)
        touches_voicing = bool(
            np.any(voiced[nearby_start:start]) or np.any(voiced[end:nearby_end])
        )
        mean_rms_db = float(np.mean(rms_db[start:end]))
        if touches_voicing:
            classification = "onset_or_offset_overhang"
        elif mean_rms_db <= rms_thresh_db:
            classification = "low_energy"
        else:
            classification = "unvoiced_high_energy"
        frame_count = end - start
        class_frames[classification] += frame_count
        regions.append({
            "start_s": float(start * hop / sr),
            "end_s": float(end * hop / sr),
            "duration_s": float(frame_count * hop / sr),
            "frame_count": frame_count,
            "rms_db_mean": mean_rms_db,
            "rms_db_peak": float(np.max(rms_db[start:end])),
            "classification": classification,
        })

    total_frames = int(np.sum(hallucinated))
    return {
        "rms_thresh_db": rms_thresh_db,
        "adjacency_s": adjacency_s,
        "region_count": len(regions),
        "total_frames": total_frames,
        "class_frames": class_frames,
        "class_ratios": {
            name: float(count / total_frames) if total_frames else 0.0
            for name, count in class_frames.items()
        },
        "regions": regions,
    }


def pitch_coverage_metrics(f0_midi: np.ndarray, midi_roll: np.ndarray) -> dict:
    n = min(len(f0_midi), len(midi_roll))
    f0_midi, midi_roll = f0_midi[:n], midi_roll[:n]
    voiced = np.isfinite(f0_midi)
    midi_active = np.isfinite(midi_roll)
    both = voiced & midi_active
    coverage = float(np.sum(both) / np.sum(voiced)) if np.sum(voiced) > 0 else 0.0
    hallucination = (float(np.sum(midi_active & ~voiced) / np.sum(midi_active))
                     if np.sum(midi_active) > 0 else 0.0)
    pitch_err = np.abs(f0_midi[both] - midi_roll[both])
    return {
        "voiced_frames": int(np.sum(voiced)),
        "midi_active_frames": int(np.sum(midi_active)),
        "co_active_frames": int(np.sum(both)),
        "coverage_ratio": coverage,
        "hallucination_ratio": hallucination,
        "pitch_error_mean_semitones": float(np.mean(pitch_err)) if len(pitch_err) else None,
        "pitch_error_median_semitones": float(np.median(pitch_err)) if len(pitch_err) else None,
        "pitch_error_p90_semitones": (
            float(np.percentile(pitch_err, 90)) if len(pitch_err) else None
        ),
        "frames_within_half_semitone_pct": (
            float(np.mean(pitch_err < 0.5)) * 100 if len(pitch_err) else 0.0
        ),
        "frames_within_one_semitone_pct": (
            float(np.mean(pitch_err < 1.0)) * 100 if len(pitch_err) else 0.0
        ),
    }


def pitch_histogram_overlap(f0_midi: np.ndarray, midi_roll: np.ndarray) -> dict:
    bins = np.arange(20, 109)
    f0_hist, _ = np.histogram(
        np.round(f0_midi[np.isfinite(f0_midi)]).astype(int), bins=bins, density=True
    )
    midi_hist, _ = np.histogram(
        np.round(midi_roll[np.isfinite(midi_roll)]).astype(int), bins=bins, density=True
    )
    return {"pitch_histogram_intersection": float(np.sum(np.minimum(f0_hist, midi_hist)))}


def note_onset_regularity(pm: pretty_midi.PrettyMIDI) -> dict:
    onsets = sorted(n.start for inst in pm.instruments for n in inst.notes)
    if len(onsets) < 2:
        return {"ioi_mean_s": None, "ioi_std_s": None, "ioi_cv": None,
                "ioi_min_s": None, "ioi_max_s": None}
    iois = np.diff(onsets)
    return {
        "ioi_mean_s": float(np.mean(iois)),
        "ioi_std_s": float(np.std(iois)),
        "ioi_cv": float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else None,
        "ioi_min_s": float(np.min(iois)),
        "ioi_max_s": float(np.max(iois)),
    }


# ---------------------------------------------------------------------------
# Workdir discovery
# ---------------------------------------------------------------------------

def resolve_from_workdir(workdir: Path) -> tuple[Path, Path, Path]:
    """Discover MIDI, vocals WAV, and mix WAV from an audio2midi output workdir."""
    midi_candidates = list(workdir.glob("*.mid"))
    if not midi_candidates:
        raise FileNotFoundError(f"No .mid file found in {workdir}")
    midi_path = midi_candidates[0]

    # Prefer MBR stem; fall back to htdemucs
    vocals_candidates = list(workdir.glob("stems/mbr/*vocals*.wav"))
    if not vocals_candidates:
        vocals_candidates = list(workdir.glob("stems/htdemucs/*/vocals.wav"))
    if not vocals_candidates:
        raise FileNotFoundError(
            f"No vocals stem found in {workdir} "
            "(checked stems/mbr/*vocals*.wav and stems/htdemucs/*/vocals.wav)"
        )
    vocals_path = vocals_candidates[0]

    mix_candidates = list(workdir.glob("extracted/*.wav"))
    if not mix_candidates:
        raise FileNotFoundError(f"No extracted/*.wav found in {workdir}")
    mix_path = mix_candidates[0]

    log.info("Workdir resolved: midi=%s  vocals=%s  mix=%s",
             midi_path.name, vocals_path.name, mix_path.name)
    return midi_path, vocals_path, mix_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vocal MIDI transcription quality evaluator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workdir", type=Path, metavar="DIR",
                        help="audio2midi output workdir; paths are auto-discovered")
    parser.add_argument("--json-out", type=Path, metavar="FILE",
                        help="also write the complete evaluation report as JSON")
    parser.add_argument("--regions-out", type=Path, metavar="FILE",
                        help="write hallucinated-frame region diagnostics as JSON")
    parser.add_argument("positional", nargs="*", metavar="FILE",
                        help="<midi_file> <vocals_wav> <mix_wav> (explicit mode)")
    args = parser.parse_args()

    if args.workdir:
        if args.positional:
            parser.error("--workdir and positional arguments are mutually exclusive")
        midi_path, vocals_path, mix_path = resolve_from_workdir(args.workdir)
    elif len(args.positional) == 3:
        midi_path, vocals_path, mix_path = (Path(p) for p in args.positional)
    else:
        parser.error("Provide --workdir <dir> or exactly three positional args: "
                     "<midi_file> <vocals_wav> <mix_wav>")

    for p in (midi_path, vocals_path, mix_path):
        if not p.exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    results = {}

    log.info("Measuring MIDI properties ...")
    results["midi_properties"] = midi_properties(midi_path)

    log.info("Loading vocals stem (%s) ...", vocals_path.name)
    vocals, sr_v = librosa.load(str(vocals_path), sr=None, mono=True)
    results["vocals_audio"] = {
        "sample_rate": int(sr_v),
        "duration_s": float(len(vocals) / sr_v),
        "peak_dBFS": float(librosa.amplitude_to_db(np.array([np.max(np.abs(vocals))]))[0]),
    }

    log.info("Loading mix (%s) ...", mix_path.name)
    mix, sr_m = librosa.load(str(mix_path), sr=None, mono=True)
    results["mix_audio"] = {
        "sample_rate": int(sr_m),
        "duration_s": float(len(mix) / sr_m),
        "peak_dBFS": float(librosa.amplitude_to_db(np.array([np.max(np.abs(mix))]))[0]),
    }

    hop = 512

    log.info("Computing vocal activity ...")
    results["vocal_activity"] = vocal_activity(vocals, sr_v, hop=hop)

    f0_hz = pyin_f0(vocals, sr_v, hop=hop)
    f0_midi_arr = f0_to_midi_pitch(f0_hz)
    voiced_pitches = f0_midi_arr[np.isfinite(f0_midi_arr)]
    results["pyin_f0"] = {
        "total_frames": len(f0_hz),
        "voiced_frames": int(np.sum(np.isfinite(f0_hz))),
        "voiced_ratio": float(np.sum(np.isfinite(f0_hz)) / len(f0_hz)),
        "f0_hz_mean": float(np.nanmean(f0_hz)) if len(voiced_pitches) > 0 else None,
        "f0_hz_median": float(np.nanmedian(f0_hz)) if len(voiced_pitches) > 0 else None,
        "f0_midi_mean": float(np.mean(voiced_pitches)) if len(voiced_pitches) > 0 else None,
        "f0_midi_median": float(np.median(voiced_pitches)) if len(voiced_pitches) > 0 else None,
        "f0_midi_min": float(np.min(voiced_pitches)) if len(voiced_pitches) > 0 else None,
        "f0_midi_max": float(np.max(voiced_pitches)) if len(voiced_pitches) > 0 else None,
        "f0_midi_range_semitones": (float(np.max(voiced_pitches) - np.min(voiced_pitches))
                                    if len(voiced_pitches) > 1 else 0),
    }

    log.info("Computing pitch coverage vs MIDI note roll ...")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    midi_roll = midi_pitch_roll(pm, sr_v, hop=hop, n_frames=len(f0_midi_arr))
    results["pitch_coverage"] = pitch_coverage_metrics(f0_midi_arr, midi_roll)
    results["hallucination_regions"] = hallucination_region_metrics(
        f0_midi_arr, midi_roll, vocals, sr_v, hop=hop
    )
    results["pitch_histogram"] = pitch_histogram_overlap(f0_midi_arr, midi_roll)
    results["onset_regularity"] = note_onset_regularity(pm)
    voiced_indices = np.flatnonzero(np.isfinite(f0_hz))
    final_voiced_s = (
        float((voiced_indices[-1] + 1) * hop / sr_v) if len(voiced_indices) else 0.0
    )
    results["pyin_f0"]["final_voiced_s"] = final_voiced_s
    results["duration_mismatch_s"] = round(
        abs(results["midi_properties"]["duration_s"] - final_voiced_s), 3
    )
    results["audio_duration_mismatch_s"] = round(
        abs(results["midi_properties"]["duration_s"] - results["vocals_audio"]["duration_s"]), 3
    )

    report_json = json.dumps(results, indent=2)
    print(report_json)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report_json + "\n")
    if args.regions_out:
        args.regions_out.parent.mkdir(parents=True, exist_ok=True)
        args.regions_out.write_text(
            json.dumps(results["hallucination_regions"], indent=2) + "\n"
        )
    log.info("Evaluation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.exception("Unhandled error: %s", exc)
        sys.exit(1)
