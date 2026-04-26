from __future__ import annotations

import argparse
import math

import numpy as np

from graded_common import FS, save_wav
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Audio interference generator for professor/evil-channel tests.")
    p.add_argument("--kind", choices=["sine", "white", "burst", "chirp"], required=True)
    p.add_argument("--freq", type=float, default=3000.0, help="sine frequency for --kind sine")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--gain", type=float, default=0.20)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav-out", default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def bandlimit_white(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(6, [20.0, 20000.0], btype="bandpass", fs=FS, output="sos")
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x.astype(np.float32)


def make_wave(kind: str, duration: float, gain: float, freq: float) -> np.ndarray:
    n = int(round(duration*FS))
    t = np.arange(n)/FS
    if kind == "sine":
        x = np.sin(2*np.pi*freq*t)
    elif kind == "white":
        rng = np.random.default_rng()
        x = bandlimit_white(rng.normal(0,1,size=n).astype(np.float32))
    elif kind == "burst":
        rng = np.random.default_rng()
        x = np.zeros(n, dtype=np.float32)
        every = int(0.7*FS)
        width = int(0.08*FS)
        for start in range(int(0.4*FS), n, every):
            stop = min(n, start+width)
            x[start:stop] += rng.normal(0, 1, stop-start).astype(np.float32)
        x = bandlimit_white(x)
    else:
        f0, f1 = 300.0, 18000.0
        k = (f1-f0)/max(duration,1e-9)
        phase = 2*np.pi*(f0*t + 0.5*k*t*t)
        x = np.sin(phase)
    x = x.astype(np.float32)
    x /= np.max(np.abs(x)) + 1e-12
    return (gain*x).astype(np.float32)


def main() -> int:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    run_dir = prepare_run_dir(label=f"interference_{args.kind}", mode=args.run_dir)
    command_manifest(run_dir, __import__('sys').argv)
    x = make_wave(args.kind, args.duration, args.gain, args.freq)
    wav = artifact_path(run_dir, f"interference_{args.kind}.wav", args.wav_out)
    if wav:
        save_wav(wav, x)
        print(f"[jammer] wrote {wav}")
    if run_dir:
        write_json(run_dir / "interference_meta.json", {"kind": args.kind, "freq": args.freq, "duration": args.duration, "gain": args.gain})
    print(f"[jammer] kind={args.kind} duration={args.duration:.1f}s gain={args.gain:.2f} freq={args.freq:.1f}")
    if args.no_play:
        return 0
    import sounddevice as sd
    sd.play(x, FS, device=args.device, blocking=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
