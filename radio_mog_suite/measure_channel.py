from __future__ import annotations

import argparse
import sys

from graded_common import (
    FS, ambient_scan, build_measurement_waveform, estimate_frequency_response, plot_ambient,
    plot_frequency_response, save_frequency_csv, save_wav,
)
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Speaker-mic measurement and ambient noise scan.")
    p.add_argument("--mode", choices=["tx", "rx", "ambient"], required=True)
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--gain", type=float, default=0.80)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    run_dir = prepare_run_dir(label=f"measure_{args.mode}", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)
    import sounddevice as sd

    if args.mode == "tx":
        wave, meta = build_measurement_waveform(args.duration)
        wave = (args.gain * wave).astype("float32")
        path = artifact_path(run_dir, "measurement_tx.wav", None)
        if path:
            save_wav(path, wave)
        print(f"[measure-tx] playing sweep duration={len(wave)/FS:.2f}s")
        sd.play(wave, FS, device=args.device, blocking=True)
        return 0

    if args.mode == "ambient":
        print(f"[ambient] recording {args.duration:.2f}s")
        x = sd.rec(int(round(args.duration*FS)), samplerate=FS, channels=1, dtype="float32", device=args.device, blocking=True)[:,0]
        rec = ambient_scan(x)
        wav = artifact_path(run_dir, "ambient_capture.wav", None)
        png = artifact_path(run_dir, "ambient_noise.png", None)
        js = artifact_path(run_dir, "ambient_choice.json", None)
        if wav: save_wav(wav, x)
        if png: plot_ambient(rec["band_powers_db"], png)
        if js: write_json(js, rec)
        print(f"[ambient] selected_profiles={','.join(rec['selected_profiles'])}")
        print(f"[ambient] notes: {rec['notes']}")
        if run_dir: print(f"[ambient] folder {run_dir.resolve()}")
        return 0

    # receiver side sweep capture
    tx_wave, _ = build_measurement_waveform(args.duration)
    seconds = len(tx_wave)/FS + 0.30
    print(f"[measure-rx] recording {seconds:.2f}s. Start TX sweep immediately.")
    x = sd.rec(int(round(seconds*FS)), samplerate=FS, channels=1, dtype="float32", device=args.device, blocking=True)[:,0]
    result = estimate_frequency_response(x, args.duration)
    wav = artifact_path(run_dir, "channel_capture.wav", None)
    png = artifact_path(run_dir, "channel_response.png", None)
    csv = artifact_path(run_dir, "channel_response.csv", None)
    js = artifact_path(run_dir, "channel_response.json", None)
    if wav: save_wav(wav, x)
    if result.get("ok"):
        freq, mag_db = result["freq"], result["mag_db"]
        if png: plot_frequency_response(freq, mag_db, png)
        if csv: save_frequency_csv(freq, mag_db, csv)
        out = {"ok": True, "score": result["score"], "recommendation": result["recommendation"]}
        if js: write_json(js, out)
        print(f"[measure-rx] OK score={result['score']:.3f}")
        print(f"[measure-rx] {result['recommendation']['notes']}")
        if run_dir: print(f"[measure-rx] folder {run_dir.resolve()}")
        return 0
    print(f"[measure-rx] FAIL {result}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
