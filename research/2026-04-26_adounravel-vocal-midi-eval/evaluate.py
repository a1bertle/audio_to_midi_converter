"""
This script has moved to scripts/eval_vocal_midi.py.

Run from the project root:
    python scripts/eval_vocal_midi.py --workdir outputs/<workdir>/
    python scripts/eval_vocal_midi.py <midi_file> <vocals_wav> <mix_wav>
"""
from pathlib import Path
import runpy
import sys

sys.argv[0] = str(Path(__file__).resolve().parents[2] / "scripts" / "eval_vocal_midi.py")
runpy.run_path(sys.argv[0], run_name="__main__")
