from __future__ import annotations

import argparse
import sys

import numpy as np

from graded_common import (
    CDMA_CARRIER, CDMA_RATE, DEFAULT_GAIN, FS, build_cdma_packet, build_fsk_packet,
    build_qpsk_packet, choose_carrier_from_file, choose_profiles_from_files, parse_profiles,
    plot_spectrum, plot_tx_waveform, prbs_bits, bitstream_hash, save_wav, play_audio,
)
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser(description="THURSDAY GRADED transmitter: uncoded/uncompressed FSK, QPSK, or CDMA PRBS stream.")
    p.add_argument("--mode", choices=["fsk", "qpsk", "cdma"], required=True)
    p.add_argument("--rate", type=float, default=None, help="bit rate. FSK/QPSK require 50/500/5000 for the graded demos. CDMA is 100 by default.")
    p.add_argument("--n-bits", type=int, default=512, help="number of uncoded payload bits in PRBS test stream")
    p.add_argument("--seed", type=int, default=1580, help="PRBS seed shared with receiver")
    p.add_argument("--profiles", default="auto", help="FSK profiles: low, high, low,high, wide, or auto")
    p.add_argument("--channel-file", default=None, help="channel_response.json for automatic FSK profile or QPSK carrier")
    p.add_argument("--ambient-file", default=None, help="ambient_choice.json for automatic FSK profile")
    p.add_argument("--carrier", type=float, default=None, help="QPSK carrier or CDMA carrier")
    p.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav-out", default=None)
    p.add_argument("--meta-out", default=None)
    p.add_argument("--preview", action="store_true", help="save TX waveform and spectrum PNGs in the run folder")
    p.add_argument("--run-dir", default="auto", help="auto, none, or explicit folder")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    run_dir = prepare_run_dir(label=f"tx_{args.mode}_{int(args.rate or 0)}bps", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)
    bits = prbs_bits(args.n_bits, args.seed)

    if args.mode == "fsk":
        rate = args.rate if args.rate is not None else 500.0
        spec = choose_profiles_from_files(args.channel_file, args.ambient_file, explicit=args.profiles)
        profiles = parse_profiles(spec)
        wave, meta = build_fsk_packet(bits, rate, profiles)
        meta.update({"profiles_spec": spec, "profiles": [p.name for p in profiles]})
    elif args.mode == "qpsk":
        rate = args.rate if args.rate is not None else 500.0
        carrier = args.carrier if args.carrier is not None else choose_carrier_from_file(args.channel_file)
        wave, meta = build_qpsk_packet(bits, rate, carrier)
    else:
        rate = args.rate if args.rate is not None else CDMA_RATE
        carrier = args.carrier if args.carrier is not None else CDMA_CARRIER
        wave, meta = build_cdma_packet(bits, rate, carrier)

    wave = np.clip(args.gain * wave, -0.98, 0.98).astype(np.float32)
    wav_path = artifact_path(run_dir, f"tx_{args.mode}_{int(meta['bit_rate_requested'])}bps.wav", args.wav_out)
    meta_path = artifact_path(run_dir, f"tx_{args.mode}_meta.json", args.meta_out)
    tx_meta = {"seed": args.seed, "n_bits": args.n_bits, "gain": args.gain, "bit_hash": bitstream_hash(bits), "first64_bits": bits[:64], **meta}

    if wav_path:
        save_wav(wav_path, wave)
        print(f"[tx] wrote wav {wav_path}")
    if meta_path:
        write_json(meta_path, tx_meta)
        print(f"[tx] wrote meta {meta_path}")
    if args.preview and run_dir is not None:
        print(f"[tx] wrote preview {plot_tx_waveform(wave, run_dir / 'tx_waveform.png')}")
        print(f"[tx] wrote spectrum {plot_spectrum(wave, run_dir / 'tx_spectrum.png')}")

    print(f"[tx] mode={args.mode} requested_rate={meta['bit_rate_requested']:.1f} actual_rate={meta.get('actual_bit_rate', meta.get('actual_rate', meta['bit_rate_requested'])):.1f} n_bits={args.n_bits} seed={args.seed} bit_hash={bitstream_hash(bits)} duration={len(wave)/FS:.3f}s gain={args.gain:.2f}")
    if args.no_play:
        return 0
    play_audio(wave, device=args.device, fs=FS)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[tx] interrupted")
        raise SystemExit(130)
