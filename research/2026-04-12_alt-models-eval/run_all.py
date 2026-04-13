"""
Alt-models benchmark script.
Research session: research/2026-04-12_alt-models-eval/
Date: 2026-04-12

Runs four separation approaches on the Foals benchmark track, then
calls evaluate_stems.py on each output and saves JSON + timing results.

Approaches:
  1. htdemucs_ft  — fine-tuned 4-stem model (drums, bass, vocals, other)
  2. htdemucs_4s  — stock 4-stem model (drums, bass, vocals, other)
  3. mdx_extra    — MDX-Net extra 4-stem model (drums, bass, vocals, other)
  4. htdemucs_6s + Wiener spectral mask — post-process current baseline
                    to sharpen guitar/other boundary

All 4-stem models produce an 'other' stem that bundles guitar+piano+etc.
For approaches 1-3, cross-leakage will be measured between the available stems.
For approach 4, the full 6-stem eval is re-run with the masked stems.
"""

from __future__ import annotations

import json
import os
import sys
import time
import shutil
import subprocess
import resource
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "inputs" / "Foals - My Number (Official Audio).mp3"
EVAL_SCRIPT = REPO_ROOT / "tests" / "evaluation" / "evaluate_stems.py"
SESSION = REPO_ROOT / "research" / "2026-04-12_alt-models-eval"
OUT_ROOT = Path("/tmp/demucs_alt_eval2")

# SSL fix (same workaround as prior benchmark session)
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# ─── helpers ─────────────────────────────────────────────────────────────────

def run_demucs(model: str, out_dir: Path, device: str = "mps") -> tuple[float, int]:
    """Run demucs separation. Returns (wall_seconds, peak_rss_bytes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", model,
        "--device", device,
        "--out", str(out_dir),
        str(INPUT),
    ]
    print(f"\n[demucs] model={model} device={device} out={out_dir}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed for {model}")
    # RSS: best approximation via resource module (post-subprocess, not perfect)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(f"[demucs] wall={wall:.1f}s  rss_approx={rss/1024/1024:.0f}MB", flush=True)
    return wall, rss


def find_stems_dir(out_dir: Path, model: str) -> Path:
    """Find the subdirectory demucs created inside out_dir."""
    # demucs creates: out_dir/<model>/<track_name>/
    for p in sorted(out_dir.rglob("*.wav")):
        return p.parent
    raise FileNotFoundError(f"No .wav files found under {out_dir}")


def run_eval(stems_dir: Path, label: str, extra_stems: list[str] | None = None) -> dict:
    """Run evaluate_stems.py and return parsed JSON."""
    json_out = SESSION / f"{label}_eval.json"
    stems_arg = extra_stems or ["vocals", "drums", "bass", "other"]
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--mix", str(INPUT),
        "--stems-dir", str(stems_dir),
        "--stems", *stems_arg,
        "--json-out", str(json_out),
    ]
    print(f"\n[eval] {label}  stems={stems_arg}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"eval failed for {label}")
    return json.loads(json_out.read_text())


def wiener_mask_guitar_other(stems_dir_6s: Path, out_dir: Path, alpha: float = 2.0) -> Path:
    """
    Post-process htdemucs_6s guitar/other stems with a Wiener-style mask.

    For each frequency bin, compute:
        M_guitar(f) = guitar(f)^alpha / (guitar(f)^alpha + other(f)^alpha + eps)
        M_other(f)  = other(f)^alpha  / (guitar(f)^alpha + other(f)^alpha + eps)
    Then apply masks to the STFTs and reconstruct via ISTFT.

    All other stems are copied unchanged.
    alpha=2.0 is a standard Wiener filter exponent [assumed from BSS literature].
    """
    import numpy as np
    import soundfile as sf

    STEMS = ["drums", "bass", "vocals", "piano"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy unaffected stems
    for s in STEMS:
        src = stems_dir_6s / f"{s}.wav"
        if src.exists():
            shutil.copy2(src, out_dir / f"{s}.wav")

    # Load guitar + other
    guitar_path = stems_dir_6s / "guitar.wav"
    other_path = stems_dir_6s / "other.wav"
    guitar_audio, sr = sf.read(str(guitar_path))
    other_audio, _ = sf.read(str(other_path))

    # Work per-channel
    n_ch = guitar_audio.shape[1] if guitar_audio.ndim > 1 else 1
    guitar_masked = np.zeros_like(guitar_audio)
    other_masked = np.zeros_like(other_audio)

    N_FFT = 4096
    HOP = N_FFT // 4

    for ch in range(n_ch):
        g = guitar_audio[:, ch] if n_ch > 1 else guitar_audio
        o = other_audio[:, ch] if n_ch > 1 else other_audio

        min_len = min(len(g), len(o))
        g, o = g[:min_len], o[:min_len]

        G = np.fft.rfft(g, n=N_FFT) if len(g) <= N_FFT else _stft_overlap_add(g, N_FFT, HOP)
        O = np.fft.rfft(o, n=N_FFT) if len(o) <= N_FFT else _stft_overlap_add(o, N_FFT, HOP)

        # Use STFT for full signal
        import librosa
        G_stft = librosa.stft(g, n_fft=N_FFT, hop_length=HOP)
        O_stft = librosa.stft(o, n_fft=N_FFT, hop_length=HOP)

        Gm = np.abs(G_stft) ** alpha
        Om = np.abs(O_stft) ** alpha
        eps = 1e-8
        denom = Gm + Om + eps

        mask_g = Gm / denom
        mask_o = Om / denom

        G_masked = G_stft * mask_g
        O_masked = O_stft * mask_o

        g_rec = librosa.istft(G_masked, hop_length=HOP, length=min_len)
        o_rec = librosa.istft(O_masked, hop_length=HOP, length=min_len)

        if n_ch > 1:
            guitar_masked[:min_len, ch] = g_rec
            other_masked[:min_len, ch] = o_rec
        else:
            guitar_masked[:min_len] = g_rec
            other_masked[:min_len] = o_rec

    sf.write(str(out_dir / "guitar.wav"), guitar_masked, sr)
    sf.write(str(out_dir / "other.wav"), other_masked, sr)
    print(f"[wiener] masked guitar+other saved to {out_dir}", flush=True)
    return out_dir


def _stft_overlap_add(x, n_fft, hop):
    """Unused placeholder — librosa.stft used directly above."""
    pass


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    SESSION.mkdir(parents=True, exist_ok=True)
    timings = {}

    # ── Approach 1: htdemucs_ft ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("APPROACH 1: htdemucs_ft (fine-tuned 4-stem, MPS)")
    print("="*60)
    out1 = OUT_ROOT / "htdemucs_ft"
    if not any(out1.rglob("*.wav")):
        wall1, rss1 = run_demucs("htdemucs_ft", out1, "mps")
        timings["htdemucs_ft"] = {"wall_s": round(wall1, 2), "rss_bytes": rss1}
    else:
        print("[skip] output already exists")
        timings["htdemucs_ft"] = {"wall_s": "cached", "rss_bytes": "n/a"}
    stems1 = find_stems_dir(out1, "htdemucs_ft")
    eval1 = run_eval(stems1, "htdemucs_ft")

    # ── Approach 2: htdemucs_4s (stock) ─────────────────────────────────────
    print("\n" + "="*60)
    print("APPROACH 2: htdemucs (stock 4-stem, MPS)")
    print("="*60)
    out2 = OUT_ROOT / "htdemucs_4s"
    if not any(out2.rglob("*.wav")):
        wall2, rss2 = run_demucs("htdemucs", out2, "mps")
        timings["htdemucs_4s"] = {"wall_s": round(wall2, 2), "rss_bytes": rss2}
    else:
        print("[skip] output already exists")
        timings["htdemucs_4s"] = {"wall_s": "cached", "rss_bytes": "n/a"}
    stems2 = find_stems_dir(out2, "htdemucs")
    eval2 = run_eval(stems2, "htdemucs_4s")

    # ── Approach 3: mdx_extra (CPU — mdx models don't support MPS) ──────────
    print("\n" + "="*60)
    print("APPROACH 3: mdx_extra (4-stem, CPU)")
    print("="*60)
    out3 = OUT_ROOT / "mdx_extra"
    if not any(out3.rglob("*.wav")):
        wall3, rss3 = run_demucs("mdx_extra", out3, "cpu")
        timings["mdx_extra"] = {"wall_s": round(wall3, 2), "rss_bytes": rss3}
    else:
        print("[skip] output already exists")
        timings["mdx_extra"] = {"wall_s": "cached", "rss_bytes": "n/a"}
    stems3 = find_stems_dir(out3, "mdx_extra")
    eval3 = run_eval(stems3, "mdx_extra")

    # ── Approach 4: htdemucs_6s + Wiener spectral masking ───────────────────
    print("\n" + "="*60)
    print("APPROACH 4: htdemucs_6s + Wiener spectral mask (alpha=2.0)")
    print("="*60)
    # Use the already-separated 6s stems from the prior research session
    stems_6s_src = Path(
        "output/2026-04-13_foals-my-number-stems/htdemucs_6s/Foals - My Number (Official Audio)"
    )
    if not stems_6s_src.is_absolute():
        stems_6s_src = REPO_ROOT / stems_6s_src

    out4 = OUT_ROOT / "htdemucs_6s_wiener"
    t4 = time.perf_counter()
    if stems_6s_src.exists():
        masked_dir = wiener_mask_guitar_other(stems_6s_src, out4, alpha=2.0)
        timings["htdemucs_6s_wiener"] = {
            "wall_s": round(time.perf_counter() - t4, 2),
            "note": "post-processing only; separation reused from prior session",
        }
    else:
        print(f"[WARN] 6s stems dir not found: {stems_6s_src}")
        print("[WARN] Re-running htdemucs_6s separation...")
        out4_sep = OUT_ROOT / "htdemucs_6s_fresh"
        wall4, rss4 = run_demucs("htdemucs_6s", out4_sep, "mps")
        stems_6s_src = find_stems_dir(out4_sep, "htdemucs_6s")
        masked_dir = wiener_mask_guitar_other(stems_6s_src, out4, alpha=2.0)
        timings["htdemucs_6s_wiener"] = {
            "wall_s": round(wall4 + (time.perf_counter() - t4), 2),
            "rss_bytes": rss4,
        }
    eval4 = run_eval(
        masked_dir, "htdemucs_6s_wiener",
        extra_stems=["vocals", "drums", "bass", "guitar", "other", "piano"]
    )

    # ── Save timing summary ──────────────────────────────────────────────────
    summary = {
        "timings": timings,
        "eval": {
            "htdemucs_ft": eval1,
            "htdemucs_4s": eval2,
            "mdx_extra": eval3,
            "htdemucs_6s_wiener": eval4,
        }
    }
    out_json = SESSION / "all_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] All results saved to {out_json}")

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n=== RESULTS COMPARISON ===")
    print(f"{'Approach':<28} {'Wall(s)':>8} {'Guitar SRR':>12} {'G/O XL(dB)':>12} {'Other SRR':>10}")
    print("-" * 74)

    rows = [
        ("htdemucs_6s (baseline)", "49.0*", "+10.57", "-3.52", "+1.83"),
    ]

    def extract(ev, guitar_key="guitar"):
        stems = ev.get("stems", {})
        g = stems.get(guitar_key, {})
        o = stems.get("other", {})
        g_srr = g.get("stem_to_residual_ratio_db", "n/a")
        o_srr = o.get("stem_to_residual_ratio_db", "n/a")
        xl = g.get("cross_leakage_db", {}).get("other", "n/a") if g else "n/a"
        return g_srr, xl, o_srr

    for label, ev, timing in [
        ("htdemucs_ft", eval1, timings.get("htdemucs_ft", {})),
        ("htdemucs_4s", eval2, timings.get("htdemucs_4s", {})),
        ("mdx_extra", eval3, timings.get("mdx_extra", {})),
        ("htdemucs_6s+wiener", eval4, timings.get("htdemucs_6s_wiener", {})),
    ]:
        g_srr, xl, o_srr = extract(ev)
        w = timing.get("wall_s", "n/a")
        print(f"{label:<28} {str(w):>8} {str(g_srr):>12} {str(xl):>12} {str(o_srr):>10}")

    print("\n* baseline wall time from research/2026-04-12_demucs-benchmark/ (95s track, MPS)")
    print("  Current track is 242.58s — wall times not directly comparable to baseline.")


if __name__ == "__main__":
    main()
