from __future__ import annotations

import argparse
import numpy as np
import sounddevice as sd

from graded_common import FS, play_audio


def parse_args():
    p = argparse.ArgumentParser(description="Quick device sanity checker for wired/wireless audio I/O.")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--freq", type=float, default=3000.0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--gain", type=float, default=0.05)
    p.add_argument("--record-seconds", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return 0

    if args.output_device is not None:
        n = int(round(args.duration * FS))
        t = np.arange(n) / FS
        x = (args.gain * np.sin(2*np.pi*args.freq*t)).astype(np.float32)
        print(f"[device-doctor] playing {args.freq:.1f} Hz to output device {args.output_device} for {args.duration:.1f}s")
        play_audio(x, device=args.output_device, fs=FS)

    if args.input_device is not None:
        print(f"[device-doctor] recording input device {args.input_device} for {args.record_seconds:.1f}s")
        rec = sd.rec(int(round(args.record_seconds*FS)), samplerate=FS, channels=1, dtype="float32", device=args.input_device, blocking=True)[:,0]
        rms = float(np.sqrt(np.mean(rec*rec)))
        peak = float(np.max(np.abs(rec)))
        print(f"[device-doctor] input rms={rms:.6f} peak={peak:.6f}")
        print("[device-doctor] if peak is near 1.0, reduce gain or use more attenuation. If rms/peak are near zero while TX is playing, wiring/device choice is wrong.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
