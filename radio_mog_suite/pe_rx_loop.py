from __future__ import annotations

import argparse
import subprocess
import sys
import json
from pathlib import Path

from run_utils import prepare_run_dir, write_json, command_manifest


def parse_args():
    p = argparse.ArgumentParser(description="Repeated receiver loop that prints running average Pe.")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--mode", choices=["fsk","qpsk","cdma"], required=True)
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--n-bits", type=int, default=512)
    p.add_argument("--seed", type=int, default=1580)
    p.add_argument("--profiles", default="auto")
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--carrier", type=float, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--run-dir", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = prepare_run_dir(label=f"pe_loop_{args.mode}_{int(args.rate)}bps", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)
    total_err = 0
    total_bits = 0
    successes = 0
    for i in range(args.count):
        sub = run_dir / f"packet_{i+1:03d}" if run_dir else "none"
        cmd = [sys.executable, "rx_graded.py", "--mode", args.mode, "--rate", str(args.rate), "--n-bits", str(args.n_bits), "--seed", str(args.seed), "--run-dir", str(sub)]
        if args.profiles: cmd += ["--profiles", args.profiles]
        if args.channel_file: cmd += ["--channel-file", args.channel_file]
        if args.ambient_file: cmd += ["--ambient-file", args.ambient_file]
        if args.carrier is not None: cmd += ["--carrier", str(args.carrier)]
        if args.timeout is not None: cmd += ["--timeout", str(args.timeout)]
        print(f"[pe-loop] receiving packet {i+1}/{args.count}")
        rc = subprocess.call(cmd)
        result_json = Path(sub) / f"rx_{args.mode}_result.json" if run_dir else None
        if result_json and result_json.exists():
            obj = json.loads(result_json.read_text())
            total_err += int(obj.get("bit_errors", args.n_bits))
            total_bits += int(obj.get("n_bits", args.n_bits))
            if obj.get("ok"):
                successes += 1
        else:
            total_err += args.n_bits
            total_bits += args.n_bits
        running = total_err / max(total_bits, 1)
        print(f"[pe-loop] running Pe={running:.6f} successes={successes}/{i+1} errors={total_err}/{total_bits}")
    summary = {"mode": args.mode, "rate": args.rate, "packets": args.count, "successes": successes, "bit_errors": total_err, "bits": total_bits, "running_pe": total_err/max(total_bits,1)}
    if run_dir: write_json(run_dir / "pe_loop_summary.json", summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
