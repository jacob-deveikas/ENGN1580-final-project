from __future__ import annotations

import argparse
import sys
import wave as wave_mod

import numpy as np

from modem_fsk_final import (
    FS,
    build_measurement_waveform,
    choose_profiles_from_ambient,
    estimate_channel_response,
    plot_ambient_psd,
    plot_frequency_response,
    save_frequency_response_csv,
)
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Channel sweep and ambient noise tools.")
    p.add_argument("--mode", required=True, choices=["tx", "rx", "ambient"])
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav", default=None)
    p.add_argument("--png-out", default=None)
    p.add_argument("--csv-out", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--f0", type=float, default=500.0)
    p.add_argument("--f1", type=float, default=10000.0)
    p.add_argument("--gain", type=float, default=0.60)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--no-play", action="store_true")
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


def main() -> int:
    args = parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    run_dir = prepare_run_dir(label=f"measure_{args.mode}", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)

    if args.mode == "tx":
        wave, meta = build_measurement_waveform(duration_s=args.duration, f0=args.f0, f1=args.f1, amplitude=args.gain)
        wav_out = artifact_path(run_dir, "measurement_tx.wav", args.wav)
        print(f"[measure-tx] duration={meta['duration_s']:.3f}s sweep={args.f0:.0f}->{args.f1:.0f}Hz")
        if wav_out:
            save_wav(wav_out, wave)
            print(f"[measure-tx] wrote {wav_out}")
        if args.no_play:
            return 0
        import sounddevice as sd
        sd.play(wave, FS, device=args.device, blocking=True)
        return 0

    if args.mode == "rx":
        if args.wav:
            recording = load_wav(args.wav)
        else:
            seconds = args.duration + 0.6
            print(f"[measure-rx] recording {seconds:.2f}s")
            import sounddevice as sd
            x = sd.rec(int(round(seconds * FS)), samplerate=FS, channels=1, dtype="float32", device=args.device, blocking=True)
            recording = x[:, 0].copy()
        result = estimate_channel_response(recording, duration_s=args.duration, f0=args.f0, f1=args.f1)
        if not result.get("ok"):
            print(f"[measure-rx] FAIL reason={result.get('reason')}")
            return 1
        png_out = artifact_path(run_dir, "channel_response.png", args.png_out)
        csv_out = artifact_path(run_dir, "channel_response.csv", args.csv_out)
        json_out = artifact_path(run_dir, "channel_response.json", args.json_out)
        plot_frequency_response(result["freq"], result["mag_db"], png_out)
        save_frequency_response_csv(result["freq"], result["mag_db"], csv_out)
        write_json(json_out, {
            "recommendation": result["recommendation"],
            "fsk_eval_low": result["fsk_eval_low"],
            "fsk_eval_high": result["fsk_eval_high"],
            "score": result["score"],
        })
        print(f"[measure-rx] OK score={result['score']:.3f}")
        print(f"[measure-rx] wrote {png_out} {csv_out} {json_out}")
        rec = result["recommendation"]
        print(f"[measure-rx] recommended FSK profile={rec['fsk_profile']} tone0={rec['fsk_tone0']:.0f}Hz tone1={rec['fsk_tone1']:.0f}Hz")
        print(f"[measure-rx] recommended QPSK carrier={rec['qpsk_fc']:.0f}Hz")
        print(f"[measure-rx] notes: {rec['notes']}")
        return 0

    if args.mode == "ambient":
        if args.wav:
            recording = load_wav(args.wav)
        else:
            print(f"[ambient] recording {args.duration:.2f}s of room noise")
            import sounddevice as sd
            x = sd.rec(int(round(args.duration * FS)), samplerate=FS, channels=1, dtype="float32", device=args.device, blocking=True)
            recording = x[:, 0].copy()
        result = choose_profiles_from_ambient(recording)
        png_out = artifact_path(run_dir, "ambient_noise.png", args.png_out)
        json_out = artifact_path(run_dir, "ambient_choice.json", args.json_out)
        plot_ambient_psd(result["band_powers_db"], png_out)
        write_json(json_out, result)
        print("[ambient] selected profiles:", ",".join(result["selected_profiles"]))
        print("[ambient] band powers (dB):", result["band_powers_db"])
        print("[ambient] notes:", result["notes"])
        print(f"[ambient] wrote {png_out} {json_out}")
        return 0

    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[measure] interrupted")
        sys.exit(130)
