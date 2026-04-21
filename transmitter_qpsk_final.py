from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from modem_qpsk_final import (
    FS,
    QPSK_AIR_REPEAT_GAP_S,
    QPSK_AIR_REPETITIONS,
    QPSK_FC_DEFAULT,
    build_qpsk_air_waveform,
    load_carrier_candidates,
    pretty_stats,
)
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transmit the QPSK / 4-QAM modem.")
    p.add_argument("message", nargs="?", default="WAKE UP, NEO")
    p.add_argument("--carrier", type=float, default=QPSK_FC_DEFAULT)
    p.add_argument("--carrier-file", default=None, help="JSON from measure_channel_final.py or auto to use ./channel_response.json")
    p.add_argument("--repetitions", type=int, default=QPSK_AIR_REPETITIONS)
    p.add_argument("--gap", type=float, default=QPSK_AIR_REPEAT_GAP_S)
    p.add_argument("--gain", type=float, default=0.75)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav-out", default=None)
    p.add_argument("--meta-out", default=None)
    p.add_argument("--run-dir", default="none")
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


def resolve_carrier_file(path: str | None) -> str | None:
    if path is None:
        return None
    if path.lower() == "auto":
        local = Path("channel_response.json")
        if local.exists():
            return str(local.resolve())
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        print(f"[warn] carrier file not found: {resolved}. Falling back to explicit/default carrier.", file=sys.stderr)
        return None
    return str(resolved)


def main() -> int:
    args = parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    run_dir = prepare_run_dir(label="qpsk_tx", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)

    carrier_file = resolve_carrier_file(args.carrier_file)
    carriers = load_carrier_candidates(args.carrier, carrier_file)
    fc = float(carriers[0]) if carriers else float(args.carrier)

    gap_samples = int(round(args.gap * FS))
    wave, meta = build_qpsk_air_waveform(args.message, fc=fc, air_repetitions=args.repetitions, gap_samples=gap_samples)
    wave = np.clip(wave * args.gain, -0.98, 0.98).astype(np.float32)

    wav_out = artifact_path(run_dir, "tx_qpsk.wav", args.wav_out)
    meta_out = artifact_path(run_dir, "tx_qpsk_meta.json", args.meta_out)
    tx_meta = {
        "message": args.message,
        "carrier_hz": fc,
        "carrier_candidates_hz": carriers,
        "repetitions": args.repetitions,
        "gap_s": args.gap,
        "gain": args.gain,
        **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, list))},
    }

    print("[tx]", pretty_stats(args.message, meta))
    print(f"[tx] repetitions={args.repetitions} gap={args.gap:.3f}s gain={args.gain:.2f} fc={fc:.0f}Hz samples={len(wave)}")

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
