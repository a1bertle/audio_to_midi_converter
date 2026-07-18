#!/usr/bin/env python3
"""Cache RMVPE's raw salience ('hidden') representation once, so a thred
sweep can call decode() repeatedly without repeating the expensive
mel2hidden forward pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio2midi.transcribers.rmvpe import _TARGET_SR, _load_rmvpe  # noqa: E402

_CHECKPOINT = Path.home() / ".cache" / "audio2midi" / "rmvpe.pt"


def main() -> None:
    vocals_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_rmvpe(_CHECKPOINT, "cpu")
    y, _ = librosa.load(str(vocals_path), sr=_TARGET_SR, mono=True)

    import torch

    audio = torch.from_numpy(y).float().to(model.device).unsqueeze(0)
    mel = model.mel_extractor(audio, center=True)
    del audio
    hidden = model.mel2hidden(mel)
    hidden = hidden.squeeze(0).cpu().numpy()

    np.save(out_dir / "rmvpe_hidden.npy", hidden)
    print(f"cached hidden {hidden.shape} to {out_dir}")


if __name__ == "__main__":
    main()
