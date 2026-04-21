from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
from pathlib import Path

from run_utils import prepare_run_dir, command_manifest, artifact_path, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated BER/PER test for the robust FSK modem.")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--message-bytes", type=int, default=8)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--profiles", default="auto")
    p.add_argument("--profiles-file", default=None)
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--gain", type=float, default=0.85)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def random_message(n_bytes: int, rng: random.Random) -> str:
    alphabet = string.ascii_uppercase + string.digits + " "
    return "".join(rng.choice(alphabet) for _ in range(n_bytes))


def bit_errors(tx: str, rx: str | None) -> tuple[int, int]:
    tx_bits = "".join(f"{b:08b}" for b in tx.encode("utf-8"))
    if rx is None:
        return len(tx_bits), len(tx_bits)
    rx_bits = "".join(f"{b:08b}" for b in rx.encode("utf-8", errors="ignore"))
    n = max(len(tx_bits), len(rx_bits))
    tx_bits = tx_bits.ljust(n, "0")
    rx_bits = rx_bits.ljust(n, "0")
    return sum(a != b for a, b in zip(tx_bits, rx_bits)), n


def main() -> int:
    args = parse_args()
    run_dir = prepare_run_dir(label="fsk_ber", mode=args.run_dir)
    command_manifest(run_dir, sys.argv)

    total_packets = 0
    good_packets = 0
    total_bit_errors = 0
    total_bits = 0
    rng = random.Random(20260419)

    for trial in range(args.trials):
        msg = random_message(args.message_bytes, rng)
        trial_dir = run_dir / f"trial_{trial:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        report = str((trial_dir / "rx_fsk_result.json").resolve())
        capture = str((trial_dir / "rx_fsk_capture.wav").resolve())
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("auto_test_fsk_final.py")),
            "--message", msg,
            "--repetitions", str(args.repetitions),
            "--profiles", args.profiles,
            "--gain", str(args.gain),
            "--run-dir", str(trial_dir),
            "--save-last", capture,
            "--json-out", report,
        ]
        if args.profiles_file is not None:
            cmd += ["--profiles-file", args.profiles_file]
        if args.channel_file is not None:
            cmd += ["--channel-file", args.channel_file]
        if args.ambient_file is not None:
            cmd += ["--ambient-file", args.ambient_file]
        if args.verbose:
            cmd.append("--verbose")
        if args.device is not None:
            cmd += ["--device", str(args.device)]

        proc = subprocess.run(cmd, capture_output=not args.verbose, text=True)
        total_packets += 1
        rx_message = None
        ok = False
        if Path(report).exists():
            with open(report, "r", encoding="utf-8") as f:
                obj = json.load(f)
            ok = bool(obj.get("ok"))
            rx_message = obj.get("message")
        if ok:
            good_packets += 1
        errs, nbits = bit_errors(msg, rx_message)
        total_bit_errors += errs
        total_bits += nbits

        if not args.verbose:
            status = "OK" if ok else "FAIL"
            print(f"[ber] trial={trial} {status} tx={msg!r} rx={rx_message!r}")

    per = 1.0 - (good_packets / max(total_packets, 1))
    ber = total_bit_errors / max(total_bits, 1)
    summary = {
        "trials": total_packets,
        "successes": good_packets,
        "packet_error_rate": per,
        "bit_error_rate": ber,
        "profiles": args.profiles,
        "repetitions": args.repetitions,
        "gain": args.gain,
    }
    write_json(run_dir / "summary.json", summary)

    print(f"[ber] successes={good_packets}/{total_packets} PER={per:.4f} BER={ber:.6f}")
    print(f"[ber] wrote {(run_dir / 'summary.json').resolve()}")
    return 0 if good_packets == total_packets else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[ber] interrupted")
        sys.exit(130)
