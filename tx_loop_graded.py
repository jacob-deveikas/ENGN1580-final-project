from __future__ import annotations

import argparse
import time

from tx_graded import main as tx_main

# Wrapper: call tx_graded.py repeatedly through subprocess-like argv manipulation is ugly.
# This script just prints the command to use for true loop when you want one process per packet.
# It is intentionally simple and robust for demos.

import subprocess, sys


def parse_args():
    p = argparse.ArgumentParser(description="Repeated uncoded packet transmitter for running Pe display.")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--gap", type=float, default=9.0)
    p.add_argument("--mode", choices=["fsk","qpsk","cdma"], required=True)
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--n-bits", type=int, default=512)
    p.add_argument("--seed", type=int, default=1580)
    p.add_argument("--profiles", default="auto")
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--carrier", type=float, default=None)
    p.add_argument("--gain", type=float, default=0.75)
    p.add_argument("--device", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base = [sys.executable, "tx_graded.py", "--mode", args.mode, "--rate", str(args.rate), "--n-bits", str(args.n_bits), "--seed", str(args.seed), "--gain", str(args.gain), "--run-dir", "none"]
    if args.profiles: base += ["--profiles", args.profiles]
    if args.channel_file: base += ["--channel-file", args.channel_file]
    if args.ambient_file: base += ["--ambient-file", args.ambient_file]
    if args.carrier is not None: base += ["--carrier", str(args.carrier)]
    if args.device is not None: base += ["--device", str(args.device)]
    for i in range(args.count):
        print(f"[tx-loop] packet {i+1}/{args.count}")
        rc = subprocess.call(base)
        if rc != 0:
            return rc
        time.sleep(args.gap)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
