from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from modem_qpsk_final import FS, QPSK_AIR_REPEAT_GAP_S, QPSK_AIR_REPETITIONS, QPSK_FC_DEFAULT, build_qpsk_air_waveform, load_carrier_candidates
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_text

WARMUP_SECONDS = 1.5
TAIL_MARGIN_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-machine auto test for the QPSK modem.")
    p.add_argument("--message", default="WAKE UP, NEO")
    p.add_argument("--carrier", type=float, default=QPSK_FC_DEFAULT)
    p.add_argument("--carrier-file", default=None, help="JSON from measure_channel_final.py or auto")
    p.add_argument("--repetitions", type=int, default=QPSK_AIR_REPETITIONS)
    p.add_argument("--gap", type=float, default=QPSK_AIR_REPEAT_GAP_S)
    p.add_argument("--gain", type=float, default=0.75)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--save-last", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--constellation-out", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def resolve_carrier_file(path: str | None) -> str | None:
    if path is None:
        return None
    if path.lower() == "auto":
        local = Path("channel_response.json")
        if local.exists():
            return str(local.resolve())
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        print(f"[warn] carrier file not found: {resolved}. Falling back to explicit/default carrier.", file=sys.stderr)
        return None
    return str(resolved)


def main() -> int:
    args = parse_args()
    run_dir = prepare_run_dir(label="qpsk_auto", mode=args.run_dir)

    carrier_file = resolve_carrier_file(args.carrier_file)
    carriers = load_carrier_candidates(args.carrier, carrier_file)
    tx_fc = carriers[0]
    gap_samples = int(round(args.gap * FS))
    air_wave, _meta = build_qpsk_air_waveform(args.message, fc=tx_fc, air_repetitions=args.repetitions, gap_samples=gap_samples)
    air_duration = len(air_wave) / float(FS)
    rx_timeout = args.timeout if args.timeout is not None else (WARMUP_SECONDS + air_duration + TAIL_MARGIN_SECONDS)

    save_last = artifact_path(run_dir, "rx_qpsk_capture.wav", args.save_last)
    json_out = artifact_path(run_dir, "rx_qpsk_result.json", args.json_out)
    constellation_out = artifact_path(run_dir, "rx_qpsk_constellation.png", args.constellation_out)
    command_manifest(run_dir, sys.argv, {"tx_fc_hz": tx_fc, "carrier_candidates_hz": carriers, "air_duration_s": air_duration})

    receiver_cmd = [
        sys.executable, "-u", str(Path(__file__).with_name("receiver_qpsk_final.py")),
        "--timeout", f"{rx_timeout:.1f}",
        "--carrier", str(tx_fc),
        "--save-last", save_last,
        "--json-out", json_out,
    ]
    if constellation_out:
        receiver_cmd += ["--constellation-out", constellation_out]
    if carrier_file:
        receiver_cmd += ["--carrier-file", carrier_file]
    if args.device is not None:
        receiver_cmd += ["--device", str(args.device)]
    if args.verbose:
        receiver_cmd.append("--verbose")

    transmitter_cmd = [
        sys.executable, "-u", str(Path(__file__).with_name("transmitter_qpsk_final.py")),
        "--repetitions", str(args.repetitions),
        "--gap", str(args.gap),
        "--gain", str(args.gain),
        "--carrier", str(tx_fc),
        args.message,
    ]
    if carrier_file:
        transmitter_cmd += ["--carrier-file", carrier_file]
    if args.device is not None:
        transmitter_cmd += ["--device", str(args.device)]

    write_text(artifact_path(run_dir, "commands.txt"), "receiver: " + " ".join(receiver_cmd) + "\n" + "transmitter: " + " ".join(transmitter_cmd) + "\n")

    print(f"[auto_test] run_dir={run_dir}")
    print(f"[auto_test] starting receiver: {' '.join(receiver_cmd)}")
    receiver = subprocess.Popen(receiver_cmd, stdout=sys.stdout, stderr=sys.stderr)

    print(f"[auto_test] waiting {WARMUP_SECONDS:.1f}s for receiver warmup...")
    time.sleep(WARMUP_SECONDS)

    print(f"[auto_test] starting transmitter: {' '.join(transmitter_cmd)}")
    transmitter = subprocess.Popen(transmitter_cmd, stdout=sys.stdout, stderr=sys.stderr)

    tx_rc = transmitter.wait()
    print(f"[auto_test] transmitter exit={tx_rc}")
    rx_rc = receiver.wait()
    print(f"[auto_test] receiver exit={rx_rc}")
    return tx_rc or rx_rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[auto_test] interrupted")
        sys.exit(130)
