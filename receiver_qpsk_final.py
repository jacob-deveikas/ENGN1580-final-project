from __future__ import annotations

import argparse
import sys
import wave as wave_mod
from pathlib import Path
from typing import Optional

import numpy as np

from modem_qpsk_final import FS, QPSK_FC_DEFAULT, decode_qpsk_from_signal, load_carrier_candidates, plot_constellation, save_json
from run_utils import artifact_path, command_manifest, prepare_run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Receive and decode the QPSK / 4-QAM modem.")
    p.add_argument("--carrier", type=float, default=QPSK_FC_DEFAULT)
    p.add_argument("--carrier-file", default=None, help="JSON from measure_channel_final.py or auto to use ./channel_response.json")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav-in", default=None)
    p.add_argument("--capture-seconds", type=float, default=None)
    p.add_argument("--save-last", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--constellation-out", default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def save_wav(path: str, wave: np.ndarray) -> None:
    x = np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave_mod.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FS)
        f.writeframes(pcm.tobytes())


def load_wav(path: str) -> np.ndarray:
    with wave_mod.open(path, "rb") as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        fs = f.getframerate()
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)
    if fs != FS:
        raise ValueError(f"wav sample rate {fs} != expected {FS}")
    if sampwidth != 2:
        raise ValueError("only 16-bit PCM wav is supported")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]
    return data


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


def print_result(tag: str, result: dict) -> None:
    if result.get("ok"):
        print(
            f"[{tag}] OK fc={result.get('fc', 0.0):.0f}Hz preamble={result.get('preamble_score', 0.0):.3f} "
            f"sync={result.get('sync_score', 0.0):.3f} bytes={result.get('payload_len', 0)}"
        )
        print(f"[{tag}] Received: {result['message']}")
    else:
        print(
            f"[{tag}] FAIL reason={result.get('reason')} fc={result.get('fc', 0.0):.0f}Hz "
            f"preamble={result.get('preamble_score', 0.0):.3f} sync={result.get('sync_score', 0.0):.3f}"
        )


def decode_data(x: np.ndarray, carriers: list[float], save_last: Optional[str], json_out: Optional[str], constellation_out: Optional[str], verbose: bool) -> int:
    if save_last:
        save_wav(save_last, x)
        if verbose:
            print(f"[rx] wrote capture to {save_last}")
    result = decode_qpsk_from_signal(x, fc=carriers[0], carriers=carriers)
    if json_out:
        save_json(result, json_out)
    if result.get("ok") and constellation_out:
        pts = result.get("constellation_corrected", result.get("constellation"))
        if pts is not None:
            plot_constellation(pts, constellation_out)
            if verbose:
                print(f"[rx] wrote constellation to {constellation_out}")
    print_result("rx", result)
    return 0 if result.get("ok") else 1


def main() -> int:
    args = parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    run_dir = prepare_run_dir(label="qpsk_rx", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)

    carrier_file = resolve_carrier_file(args.carrier_file)
    carriers = load_carrier_candidates(args.carrier, carrier_file)

    save_last = artifact_path(run_dir, "rx_qpsk_capture.wav", args.save_last)
    json_out = artifact_path(run_dir, "rx_qpsk_result.json", args.json_out)
    constellation_out = artifact_path(run_dir, "rx_qpsk_constellation.png", args.constellation_out)

    if args.wav_in:
        x = load_wav(args.wav_in)
        return decode_data(x, carriers, save_last, json_out, constellation_out, args.verbose)

    seconds = args.capture_seconds if args.capture_seconds is not None else args.timeout
    print(f"[rx] recording {seconds:.2f}s then decoding ...")
    import sounddevice as sd
    x = sd.rec(int(round(seconds * FS)), samplerate=FS, channels=1, dtype="float32", device=args.device, blocking=True)
    return decode_data(x[:, 0].copy(), carriers, save_last, json_out, constellation_out, args.verbose)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[rx] interrupted")
        sys.exit(130)
