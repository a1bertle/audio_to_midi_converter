# evaluate_stems.py

Blind (no reference track required) quality evaluator for stem separation
outputs. Measures how well a source-separation tool has isolated each
instrument stem, and how much bleed-through remains from other instruments.

---

## Usage

```bash
python tests/evaluation/evaluate_stems.py \
    --mix  "inputs/Foals - My Number (Official Audio).mp3" \
    --stems-dir "output/2026-04-13_foals-my-number-stems/htdemucs_6s/Foals - My Number (Official Audio)" \
    [--stems vocals drums bass guitar piano other] \
    [--json-out output/my-run/evaluation.json] \
    [--sample-rate 44100]
```

`--stems-dir` must contain `<stem_name>.wav` files (one per stem).  
`--mix` is the original full mix — required for reconstruction metrics.  
`--json-out` writes a machine-readable report alongside the console output.

---

## Metrics explained

### Reconstruction metrics (whole-mix level)

These tell you how faithfully the stems reconstruct the original mix when
summed back together.

#### `leakage_ratio_db`

```
leakage_ratio_db = 10 * log10(energy(mix − sum(stems)) / energy(mix))
```

The **residual** is what's left when you subtract the sum of all stems from
the original mix. A large negative value (e.g. −20 dB) means the stems
account for almost all of the mix's energy — good reconstruction. A value
close to 0 dB means a lot of signal was dropped or distorted during
separation.

| Value | Meaning |
|-------|---------|
| −30 dB or lower | Excellent reconstruction |
| −20 dB | Good — ~1% of mix energy unaccounted for |
| −10 dB | Mediocre — ~10% of mix energy lost |
| 0 dB or higher | Poor — stems barely reconstruct the mix |

---

### Per-stem metrics

Reported for each stem individually.

#### `rms_db`

Root-mean-square level of the stem in dBFS (decibels relative to full
scale). Gives a rough sense of how loud/present the stem is.

A value near −70 dBFS with a very high `silence_fraction` means the model
essentially produced silence for that stem — it found no content of that
type in the mix (e.g. a `piano` stem from a track with no piano).

#### `stem_to_residual_ratio_db` (SRR)

```
SRR = 10 * log10(energy(stem) / energy(residual))
```

Compares the stem's energy to the leftover reconstruction error. A high
positive value means the stem carries a lot of real signal relative to the
artefacts. A negative value means the stem is quieter than the residual
noise floor — almost certainly empty or heavily over-suppressed.

| Value | Meaning |
|-------|---------|
| +20 dB or higher | Very clean isolation |
| +10–20 dB | Acceptable — some bleed but stem dominates |
| 0–10 dB | Marginal — bleed is significant |
| Negative | Stem is mostly noise/silence |

This is the **primary quality indicator** for a single stem. Use it to
compare techniques: higher SRR on the stem you care about = cleaner
isolation.

#### `cross_leakage_db`

```
cross_leakage_db(A, B) = 20 * log10(spectral_cosine_similarity(A, B))
```

Spectral cosine similarity measures how similar the frequency-magnitude
profiles of two stems are. It is then expressed in dB. A value close to
0 dB means the two stems share almost the same spectral shape — strong
bleed-through. A large negative value means they are spectrally orthogonal
— well isolated from each other.

This directly quantifies the "faint background instrument" problem: if
`guitar` vs `other` shows −3.5 dB, roughly 67% of the guitar's spectral
energy is shared with `other`, which is why you can hear the guitar faintly
in the `other` stem and vice versa.

| Value | Meaning |
|-------|---------|
| −30 dB or lower | Well isolated — little shared spectral shape |
| −20 dB | Moderate isolation |
| −10 dB | Noticeable bleed |
| −5 dB or higher | Heavy bleed — stems sound very similar |

#### `spectral_centroid_hz`

Mean spectral centroid of the stem across all frames. A rough indicator of
the stem's tonal character: low values (~200–500 Hz) suggest bass/kick
content, high values (>3000 Hz) suggest cymbals, highs, or noise artefacts.
Useful as a sanity check — e.g. a `bass` stem with a centroid of 4000 Hz
is suspicious.

#### `silence_fraction`

Fraction of short frames (512 samples) where the stem's RMS is below
−60 dBFS. A high value (>0.5) means the model has heavily suppressed or
blanked most of the track — either the instrument isn't present, or the
model is over-aggressive. A value near 0 means the stem is active
throughout.

---

## How to compare two techniques

Run the evaluator against each technique's output directory, saving a JSON
report for each:

```bash
python tests/evaluation/evaluate_stems.py \
    --mix "inputs/Foals - My Number (Official Audio).mp3" \
    --stems-dir output/technique-A/stems/ \
    --json-out output/technique-A/evaluation.json

python tests/evaluation/evaluate_stems.py \
    --mix "inputs/Foals - My Number (Official Audio).mp3" \
    --stems-dir output/technique-B/stems/ \
    --json-out output/technique-B/evaluation.json
```

Then compare `stem_to_residual_ratio_db` and `cross_leakage_db` for the
stems you care about. Higher SRR and more-negative cross-leakage = cleaner
separation.

---

## Limitations

These are **blind metrics** — they require no ground-truth reference stems.
That makes them easy to run on any real-world track, but they have limits:

- **Cross-leakage is spectral, not temporal.** Two stems that are active at
  different times but share the same pitch range will show high
  cross-leakage even if the separation is actually clean.
- **SRR depends on residual quality.** If the model reconstructs the mix
  poorly overall (high `leakage_ratio_db`), SRR values for all stems will
  be inflated.
- **No absolute quality threshold.** The numbers are most useful
  comparatively (technique A vs technique B) rather than as absolute
  pass/fail criteria. Subjective listening remains the final arbiter.

For a reference-based evaluation (SDR/SIR/SAR per MUSDB18 convention),
`museval` would be needed, but that requires isolated ground-truth stems
which are typically unavailable for real-world tracks.
