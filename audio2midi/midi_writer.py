"""MIDI read/write helpers."""

from __future__ import annotations

from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, tick2second

from audio2midi.models import Instrument, NoteEvent, PedalEvent, TranscriptionResult

# General MIDI program numbers (0-indexed).
_INSTRUMENT_PROGRAM: dict[Instrument, int] = {
    Instrument.PIANO: 0,     # Acoustic Grand Piano
    Instrument.GUITAR: 25,   # Acoustic Guitar (Steel)
    Instrument.VOCALS: 53,   # Voice Oohs
}


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo: int) -> int:
    return int(round((seconds * 1_000_000 * ticks_per_beat) / tempo))


def write_midi_file(
    result: TranscriptionResult,
    output_path: Path,
    bpm: float = 120.0,
    ticks_per_beat: int = 480,
) -> Path:
    """Write transcription result to a MIDI file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)
    tempo = bpm2tempo(bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    program = _INSTRUMENT_PROGRAM.get(result.instrument, 0)
    track.append(Message("program_change", program=program, time=0))

    events: list[tuple[int, int, Message]] = []
    for note in result.notes:
        start_tick = _seconds_to_ticks(
            note.start,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        end_tick = _seconds_to_ticks(
            note.end,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        velocity = max(1, min(127, int(note.velocity)))
        bounded_pitch = max(0, min(127, note.pitch))
        events.append(
            (
                start_tick,
                2,
                Message(
                    "note_on",
                    note=bounded_pitch,
                    velocity=velocity,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_tick,
                1,
                Message("note_off", note=bounded_pitch, velocity=0, time=0),
            )
        )
    for pedal in result.pedals:
        start_tick = _seconds_to_ticks(
            pedal.start,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        end_tick = _seconds_to_ticks(
            pedal.end,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        on_val = max(0, min(127, pedal.value))
        events.append(
            (
                start_tick,
                0,
                Message("control_change", control=64, value=on_val, time=0),
            )
        )
        events.append(
            (
                end_tick,
                0,
                Message("control_change", control=64, value=0, time=0),
            )
        )

    events.sort(key=lambda item: (item[0], item[1]))
    last_tick = 0
    for tick, _priority, msg in events:
        delta = max(0, tick - last_tick)
        msg.time = delta
        track.append(msg)
        last_tick = tick
    midi.save(str(output_path))
    return output_path


def read_midi_as_transcription(midi_path: Path) -> TranscriptionResult:
    """Parse MIDI file into note and sustain pedal events."""
    midi = MidiFile(str(midi_path))
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000
    notes: list[NoteEvent] = []
    pedals: list[PedalEvent] = []
    active_notes: dict[int, tuple[float, int]] = {}
    active_pedal_start: float | None = None
    now_ticks = 0

    merged = MidiTrack()
    for track in midi.tracks:
        tick_cursor = 0
        for msg in track:
            tick_cursor += msg.time
            merged.append(msg.copy(time=tick_cursor))
    merged.sort(key=lambda msg: msg.time)

    for msg in merged:
        now_ticks = msg.time
        now_seconds = tick2second(now_ticks, ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = (now_seconds, msg.velocity)
            continue
        if msg.type in {"note_off", "note_on"} and msg.note in active_notes:
            start, velocity = active_notes.pop(msg.note)
            notes.append(
                NoteEvent(
                    pitch=msg.note,
                    start=float(start),
                    end=float(now_seconds),
                    velocity=int(velocity),
                )
            )
            continue
        if msg.type == "control_change" and msg.control == 64:
            if msg.value >= 64 and active_pedal_start is None:
                active_pedal_start = float(now_seconds)
            elif msg.value < 64 and active_pedal_start is not None:
                pedals.append(
                    PedalEvent(
                        start=active_pedal_start,
                        end=float(now_seconds),
                        value=127,
                    )
                )
                active_pedal_start = None

    max_time = tick2second(now_ticks, ticks_per_beat, tempo)
    for note, (start, velocity) in active_notes.items():
        notes.append(
            NoteEvent(
                pitch=note,
                start=float(start),
                end=float(max_time),
                velocity=int(velocity),
            )
        )
    if active_pedal_start is not None:
        pedals.append(
            PedalEvent(
                start=active_pedal_start,
                end=float(max_time),
                value=127,
            )
        )
    return TranscriptionResult(notes=notes, pedals=pedals)
