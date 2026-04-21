from __future__ import annotations

import argparse
import sys
import wave as wave_mod
from pathlib import Path
from typing import Optional

import numpy as np

from modem_fsk_final import FS, decode_best_from_signal, resolve_profile_spec, resolve_profiles
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Receive and decode the robust FSK modem.")
    p.add_argument("--scheme", default="fsk")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav-in", default=None)
    p.add_argument("--capture-seconds", type=float, default=None)
    p.add_argument("--save-last", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--profiles", default="auto", help="low, high, low,high, or auto")
    p.add_argument("--profiles-file", default=None)
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--run-dir", default="auto", help="auto, none, or explicit folder path")
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


def print_result(tag: str, result: dict) -> None:
    if result.get("ok"):
        print(
            f"[{tag}] OK profile={result.get('profile', '?')} preamble={result.get('preamble_score', 0.0):.3f} "
            f"sync={result.get('sync_score', 0.0):.3f} bytes={result.get('payload_len', 0)}"
        )
        print(f"[{tag}] Received: {result['message']}")
    else:
        print(
            f"[{tag}] FAIL reason={result.get('reason')} profile={result.get('profile', '?')} "
            f"preamble={result.get('preamble_score', 0.0):.3f} sync={result.get('sync_score', 0.0):.3f}"
        )


def decode_data(x: np.ndarray, allowed_profiles, save_last: Optional[str], json_out: Optional[str], verbose: bool) -> int:
    if save_last:
        save_wav(save_last, x)
        if verbose:
            print(f"[rx] wrote capture to {save_last}")
    result = decode_best_from_signal(x, search_tail=False, allowed_profiles=allowed_profiles)
    if json_out:
        write_json(json_out, {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool))})
    print_result("rx", result)
    return 0 if result.get("ok") else 1


def main() -> int:
    args = parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if args.scheme.lower() != "fsk":
        raise ValueError("this receiver currently supports --scheme fsk only")

    run_dir = prepare_run_dir(label="fsk_rx", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)

    profiles_spec = resolve_profile_spec(
        explicit=args.profiles,
        profiles_file=args.profiles_file,
        channel_file=args.channel_file,
        ambient_file=args.ambient_file,
    )
    allowed_profiles = resolve_profiles(profiles_spec)

    save_last = artifact_path(run_dir, "rx_fsk_capture.wav", args.save_last)
    json_out = artifact_path(run_dir, "rx_fsk_result.json", args.json_out)

    if args.wav_in:
        x = load_wav(args.wav_in)
        return decode_data(x, allowed_profiles, save_last, json_out, args.verbose)

    seconds = args.capture_seconds if args.capture_seconds is not None else args.timeout
    print(f"[rx] recording {seconds:.2f}s then decoding ...")
    import sounddevice as sd
    x = sd.rec(int(round(seconds * FS)), samplerate=FS, channels=1, dtype="float32", device=args.device, blocking=True)
    return decode_data(x[:, 0].copy(), allowed_profiles, save_last, json_out, args.verbose)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[rx] interrupted")
        sys.exit(130)
