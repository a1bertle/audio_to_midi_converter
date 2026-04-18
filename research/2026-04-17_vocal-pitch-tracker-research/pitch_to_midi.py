"""Shared pitch track → MIDI note conversion utility.

Input:  f0 array (Hz), times array (seconds), optional confidence array
Output: mido.MidiFile with note events on channel 0
"""
import numpy as np
import mido


def f0_to_midi_note(f0_hz: float) -> int:
    """Convert Hz to nearest MIDI note number."""
    return int(round(12 * np.log2(max(f0_hz, 1e-6) / 440.0) + 69))


def pitch_track_to_midi(
    times: np.ndarray,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    confidence: np.ndarray | None = None,
    min_note_duration_s: float = 0.08,
    velocity_scale: float = 80.0,
    bpm: float = 120.0,
    onset_times: np.ndarray | None = None,
) -> mido.MidiFile:
    """Convert a frame-level pitch track to a mido MidiFile.

    Args:
        times:              Frame timestamps in seconds (N,)
        f0_hz:              F0 estimates in Hz (N,); 0.0 or NaN = unvoiced
        voiced:             Boolean voiced flag (N,)
        confidence:         Optional per-frame confidence (N,); used for velocity
        min_note_duration_s: Drop notes shorter than this threshold
        velocity_scale:     MIDI velocity assigned when confidence == 1.0
        bpm:                Tempo written into the MIDI file
    Returns:
        mido.MidiFile with a single track on channel 0
    """
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    if confidence is None:
        confidence = np.ones_like(f0_hz, dtype=float)

    # Snap f0 to semitones upfront so note boundaries are stable
    midi_track = np.where(
        voiced & (f0_hz > 0) & ~np.isnan(f0_hz),
        np.round(12 * np.log2(np.maximum(f0_hz, 1e-6) / 440.0) + 69).astype(int),
        -1,
    )

    notes: list[tuple[float, float, int, int]] = []  # (start_s, end_s, midi_note, velocity)
    in_note = False
    note_start = 0.0
    note_midi = 0
    conf_sum = 0.0
    frame_count = 0

    for t, midi, conf in zip(times, midi_track, confidence):
        if midi >= 0:
            if not in_note:
                in_note = True
                note_start = float(t)
                note_midi = int(midi)
                conf_sum = float(conf)
                frame_count = 1
            elif int(midi) != note_midi:
                duration = float(t) - note_start
                if duration >= min_note_duration_s:
                    vel = max(1, min(127, int((conf_sum / frame_count) * velocity_scale)))
                    notes.append((note_start, float(t), note_midi, vel))
                note_start = float(t)
                note_midi = int(midi)
                conf_sum = float(conf)
                frame_count = 1
            else:
                conf_sum += float(conf)
                frame_count += 1
        else:
            if in_note:
                duration = float(t) - note_start
                if duration >= min_note_duration_s:
                    vel = max(1, min(127, int((conf_sum / frame_count) * velocity_scale)))
                    notes.append((note_start, float(t), note_midi, vel))
                in_note = False

    # Merge short notes into their longer neighbours.
    # A note shorter than merge_threshold that is sandwiched between two notes
    # of the same pitch gets absorbed into them (gap fill).
    # Any remaining note shorter than min_note_duration_s * 2 that is within
    # 2 semitones of its neighbour gets its pitch replaced by the neighbour's.
    def _merge_notes(note_list, gap_s=0.05, proximity_semitones=2, max_silence_s=0.04):
        """Merge short ornament notes into neighbours, but never across a silence gap.

        max_silence_s: if the gap between two consecutive notes exceeds this,
        they are treated as separate phrases and never merged.
        """
        if len(note_list) <= 1:
            return note_list
        merged = list(note_list)
        changed = True
        while changed:
            changed = False
            out = []
            i = 0
            while i < len(merged):
                if i + 2 < len(merged):
                    s0, e0, n0, v0 = merged[i]
                    s1, e1, n1, v1 = merged[i + 1]
                    s2, e2, n2, v2 = merged[i + 2]
                    gap_01 = s1 - e0  # silence between note 0 and 1
                    gap_12 = s2 - e1  # silence between note 1 and 2
                    # only merge if notes are adjacent (no real silence between them)
                    if (gap_01 <= max_silence_s and gap_12 <= max_silence_s
                            and n0 == n2 and (e1 - s1) < gap_s):
                        out.append((s0, e2, n0, max(v0, v1, v2)))
                        i += 3
                        changed = True
                        continue
                if i + 1 < len(merged):
                    s0, e0, n0, v0 = merged[i]
                    s1, e1, n1, v1 = merged[i + 1]
                    gap_01 = s1 - e0
                    if (gap_01 <= max_silence_s
                            and (e1 - s1) < gap_s
                            and abs(n1 - n0) <= proximity_semitones):
                        out.append((s0, e1, n0, max(v0, v1)))
                        i += 2
                        changed = True
                        continue
                out.append(merged[i])
                i += 1
            merged = out
        return merged

    notes = _merge_notes(notes, gap_s=min_note_duration_s * 1.5,
                         max_silence_s=min_note_duration_s * 0.5)

    # Split long same-pitch notes at onset boundaries (multi-syllabic recovery).
    # Only split if the onset falls within the note (not at the very start/end)
    # and the resulting fragments would both be >= min_note_duration_s.
    if onset_times is not None and len(onset_times) > 0:
        onset_set = np.sort(onset_times)
        split_notes = []
        for start_s, end_s, note, vel in notes:
            # find onsets strictly inside this note, with margin
            margin = min_note_duration_s * 0.5
            interior = onset_set[(onset_set > start_s + margin) &
                                  (onset_set < end_s - margin)]
            if len(interior) == 0:
                split_notes.append((start_s, end_s, note, vel))
            else:
                # split at each interior onset
                boundaries = [start_s] + list(interior) + [end_s]
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    if b - a >= min_note_duration_s:
                        split_notes.append((a, b, note, vel))
        notes = split_notes

    ticks_per_s = 480 * (bpm / 60.0)
    events: list[tuple[int, mido.Message]] = []
    for start_s, end_s, note, vel in notes:
        events.append((int(start_s * ticks_per_s),
                       mido.Message("note_on", note=note, velocity=vel, time=0)))
        events.append((int(end_s * ticks_per_s),
                       mido.Message("note_off", note=note, velocity=0, time=0)))

    events.sort(key=lambda x: x[0])
    current_tick = 0
    for abs_tick, msg in events:
        msg.time = abs_tick - current_tick
        track.append(msg)
        current_tick = abs_tick

    return mid
