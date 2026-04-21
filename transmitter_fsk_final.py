from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from modem_fsk_final import (
    AIR_REPEAT_GAP_S,
    AIR_REPETITIONS,
    DEFAULT_TX_GAIN,
    FS,
    build_air_waveform_selected,
    pretty_stats,
    resolve_profile_spec,
    resolve_profiles,
)
from run_utils import artifact_path, command_manifest, prepare_run_dir, resolve_optional_path, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transmit the robust FSK modem.")
    p.add_argument("message", nargs="?", default="WAKE UP, NEO")
    p.add_argument("--scheme", default="fsk")
    p.add_argument("--repetitions", type=int, default=AIR_REPETITIONS)
    p.add_argument("--gap", type=float, default=AIR_REPEAT_GAP_S)
    p.add_argument("--profiles", default="auto", help="low, high, low,high, or auto")
    p.add_argument("--profiles-file", default=None)
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav-out", default=None)
    p.add_argument("--meta-out", default=None)
    p.add_argument("--gain", type=float, default=DEFAULT_TX_GAIN)
    p.add_argument("--run-dir", default="none", help="auto, none, or explicit folder path")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--no-play", action="store_true")
    return p.parse_args()


def maybe_write_wav(path: str, wave: np.ndarray) -> None:
    import wave as wave_mod

    scaled = np.clip(wave, -1.0, 1.0)
    pcm = (scaled * 32767.0).astype(np.int16)
    with wave_mod.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FS)
        f.writeframes(pcm.tobytes())


def main() -> int:
    args = parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if args.scheme.lower() != "fsk":
        raise ValueError("this transmitter currently supports --scheme fsk only")

    run_dir = prepare_run_dir(label="fsk_tx", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)

    profiles_spec = resolve_profile_spec(
        explicit=args.profiles,
        profiles_file=args.profiles_file,
        channel_file=args.channel_file,
        ambient_file=args.ambient_file,
    )
    selected = resolve_profiles(profiles_spec)
    gap_samples = int(round(args.gap * FS))
    wave, meta = build_air_waveform_selected(args.message, selected, air_repetitions=args.repetitions, gap_samples=gap_samples)
    wave = np.clip(wave * args.gain, -0.98, 0.98).astype(np.float32)

    wav_out = artifact_path(run_dir, "tx_fsk.wav", args.wav_out)
    meta_out = artifact_path(run_dir, "tx_fsk_meta.json", args.meta_out)

    tx_meta = {
        "message": args.message,
        "profiles_spec": profiles_spec,
        "profiles": [p.name for p in selected],
        "repetitions": args.repetitions,
        "gap_s": args.gap,
        "gain": args.gain,
        **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, list))},
    }

    print("[tx]", pretty_stats(args.message, meta))
    print(f"[tx] repetitions={args.repetitions} gap={args.gap:.3f}s gain={args.gain:.2f} profiles={profiles_spec} samples={len(wave)}")

    if wav_out:
        maybe_write_wav(wav_out, wave)
        print(f"[tx] wrote {wav_out}")
    if meta_out:
        write_json(meta_out, tx_meta)

    if args.no_play:
        return 0

    import sounddevice as sd
    sd.play(wave, FS, device=args.device, blocking=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[tx] interrupted")
        sys.exit(130)
