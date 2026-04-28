from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from run_utils import command_manifest, prepare_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser(description="One-machine auto test for THURSDAY GRADED uncoded modes.")
    p.add_argument("--mode", choices=["fsk", "qpsk", "cdma"], required=True)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--n-bits", type=int, default=512)
    p.add_argument("--seed", type=int, default=1580)
    p.add_argument("--profiles", default="auto")
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--carrier", type=float, default=None)
    p.add_argument("--gain", type=float, default=0.75)
    p.add_argument("--warmup", type=float, default=1.5)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def default_rate(mode: str, rate):
    if rate is not None:
        return rate
    return 100.0 if mode == "cdma" else 500.0


def timeout_for(mode: str, rate: float, n_bits: int, warmup: float) -> float:
    if mode == "qpsk":
        return max(7.0, warmup + (n_bits + 192) / max(rate, 1) + 4.0)
    return max(10.0, warmup + (n_bits + 48) / max(rate, 1) + 4.0)


def main() -> int:
    args = parse_args()
    rate = default_rate(args.mode, args.rate)
    run_dir = prepare_run_dir(label=f"auto_{args.mode}_{int(rate)}bps", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)
    timeout = args.timeout if args.timeout is not None else timeout_for(args.mode, rate, args.n_bits, args.warmup)

    rx_cmd = [sys.executable, "-u", "rx_graded.py", "--mode", args.mode, "--rate", str(rate), "--n-bits", str(args.n_bits), "--seed", str(args.seed), "--timeout", str(timeout)]
    tx_cmd = [sys.executable, "-u", "tx_graded.py", "--mode", args.mode, "--rate", str(rate), "--n-bits", str(args.n_bits), "--seed", str(args.seed), "--gain", str(args.gain), "--run-dir", "none"]
    if run_dir is not None:
        rx_cmd += ["--run-dir", str(run_dir / "rx")]
        tx_cmd += ["--wav-out", str(run_dir / f"tx_{args.mode}.wav"), "--meta-out", str(run_dir / f"tx_{args.mode}_meta.json"), "--preview"]
        if args.mode == "qpsk":
            rx_cmd += ["--constellation-out", str(run_dir / "qpsk_constellation.png")]
    else:
        rx_cmd += ["--run-dir", "none"]
    if args.verbose:
        rx_cmd.append("--verbose")
    for flag, val in [("--profiles", args.profiles), ("--channel-file", args.channel_file), ("--ambient-file", args.ambient_file)]:
        if val:
            rx_cmd += [flag, val]
            tx_cmd += [flag, val]
    if args.carrier is not None:
        rx_cmd += ["--carrier", str(args.carrier)]
        tx_cmd += ["--carrier", str(args.carrier)]

    print("[auto] receiver:", " ".join(rx_cmd))
    rx = subprocess.Popen(rx_cmd)
    print(f"[auto] waiting {args.warmup:.2f}s for receiver warmup")
    time.sleep(args.warmup)
    print("[auto] transmitter:", " ".join(tx_cmd))
    tx = subprocess.Popen(tx_cmd)
    tx_rc = tx.wait()
    print(f"[auto] transmitter exit={tx_rc}")
    rx_rc = rx.wait()
    print(f"[auto] receiver exit={rx_rc}")
    if run_dir is not None:
        write_json(run_dir / "auto_result.json", {"tx_exit": tx_rc, "rx_exit": rx_rc, "mode": args.mode, "rate": rate, "n_bits": args.n_bits})
        print(f"[auto] run folder {run_dir.resolve()}")
    return tx_rc or rx_rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[auto] interrupted")
        raise SystemExit(130)
