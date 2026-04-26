from __future__ import annotations

import argparse
import sys

import numpy as np

from graded_common import (
    CDMA_CARRIER, CDMA_RATE, FS, choose_carrier_from_file, choose_profiles_from_files,
    decode_cdma_capture, decode_fsk_capture, decode_qpsk_capture, load_wav, parse_profiles,
    plot_constellation, save_wav,
)
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser(description="THURSDAY GRADED receiver: uncoded Pe measurement for FSK, QPSK, or CDMA.")
    p.add_argument("--mode", choices=["fsk", "qpsk", "cdma"], required=True)
    #p.add_argument("--mode", 'fsk', required=True)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--n-bits", type=int, default=512)
    p.add_argument("--seed", type=int, default=1580)
    p.add_argument("--profiles", default="auto")
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--carrier", type=float, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--wav-in", default=None)
    p.add_argument("--save-capture", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--constellation-out", default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def print_result(r: dict) -> None:
    if r.get("ok"):
        print(f"[rx] OK mode={r.get('mode')} pe={r.get('pe', 1):.6f} errors={r.get('bit_errors')}/{r.get('n_bits')} sync={r.get('sync_score', r.get('score', 0)):.3f}")
    else:
        print(f"[rx] FAIL reason={r.get('reason')} mode={r.get('mode')} pe={r.get('pe', 1):.6f} errors={r.get('bit_errors', '?')}/{r.get('n_bits', '?')} sync={r.get('sync_score', r.get('score', 0)):.3f}")
    if "profile" in r:
        print(f"[rx] profile={r['profile']}")
    if "carrier" in r:
        print(f"[rx] carrier={r['carrier']}")


def estimate_timeout(mode: str, rate: float, n_bits: int) -> float:
    if mode == "qpsk":
        return max(5.0, 0.5 + n_bits/max(rate,1) + 3.0)
    if mode == "cdma":
        return max(8.0, 0.5 + (n_bits + 48)/max(rate,1) + 3.0)
    return max(8.0, 0.5 + (n_bits + 48)/max(rate,1) + 3.0)


def main() -> int:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    rate = args.rate
    if rate is None:
        rate = 500.0 if args.mode != "cdma" else CDMA_RATE
    run_dir = prepare_run_dir(label=f"rx_{args.mode}_{int(rate)}bps", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)
    capture_path = artifact_path(run_dir, f"rx_{args.mode}_capture.wav", args.save_capture)
    json_path = artifact_path(run_dir, f"rx_{args.mode}_result.json", args.json_out)
    const_path = artifact_path(run_dir, "qpsk_constellation.png", args.constellation_out)

    if args.wav_in:
        x = load_wav(args.wav_in)
    else:
        seconds = args.timeout if args.timeout is not None else estimate_timeout(args.mode, rate, args.n_bits)
        print(f"[rx] recording {seconds:.2f}s then decoding ...")
        import sounddevice as sd
        rec = sd.rec(int(round(seconds*FS)), samplerate=FS, channels=1, dtype="float32", device=args.device, blocking=True)
        x = rec[:,0].copy()
    if capture_path:
        save_wav(capture_path, x)
        if args.verbose:
            print(f"[rx] wrote capture {capture_path}")

    if args.mode == "fsk":
        spec = choose_profiles_from_files(args.channel_file, args.ambient_file, explicit=args.profiles)
        profiles = parse_profiles(spec)
        result = decode_fsk_capture(x, rate, args.n_bits, profiles, seed=args.seed)
        result["profiles_spec"] = spec
    elif args.mode == "qpsk":
        carrier = args.carrier if args.carrier is not None else choose_carrier_from_file(args.channel_file)
        result = decode_qpsk_capture(x, rate, args.n_bits, carrier, seed=args.seed, search=True)
        if const_path and "constellation" in result:
            plot_constellation(result["constellation"], const_path, title=f"QPSK Constellation at {rate:.0f} bps")
            print(f"[rx] wrote constellation {const_path}")
    else:
        carrier = args.carrier if args.carrier is not None else CDMA_CARRIER
        result = decode_cdma_capture(x, args.n_bits, rate=rate, carrier=carrier, seed=args.seed)

    scrub = {k:v for k,v in result.items() if k not in {"constellation", "sync_constellation", "received_bits", "expected_bits"}}
    if json_path:
        write_json(json_path, scrub)
        if args.verbose:
            print(f"[rx] wrote result {json_path}")
    print_result(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[rx] interrupted")
        raise SystemExit(130)
