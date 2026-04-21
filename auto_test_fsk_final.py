from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from modem_fsk_final import AIR_REPEAT_GAP_S, AIR_REPETITIONS, DEFAULT_TX_GAIN, FS, build_air_waveform_selected, resolve_profile_spec, resolve_profiles
from run_utils import artifact_path, command_manifest, prepare_run_dir, write_text

WARMUP_SECONDS = 1.5
TAIL_MARGIN_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-machine auto test for the robust FSK modem.")
    p.add_argument("--message", default="WAKE UP, NEO")
    p.add_argument("--repetitions", type=int, default=AIR_REPETITIONS)
    p.add_argument("--gap", type=float, default=AIR_REPEAT_GAP_S)
    p.add_argument("--profiles", default="auto")
    p.add_argument("--profiles-file", default=None)
    p.add_argument("--channel-file", default=None)
    p.add_argument("--ambient-file", default=None)
    p.add_argument("--gain", type=float, default=DEFAULT_TX_GAIN)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--run-dir", default="auto")
    p.add_argument("--save-last", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = prepare_run_dir(label="fsk_auto", mode=args.run_dir)

    profiles_spec = resolve_profile_spec(
        explicit=args.profiles,
        profiles_file=args.profiles_file,
        channel_file=args.channel_file,
        ambient_file=args.ambient_file,
    )
    selected = resolve_profiles(profiles_spec)
    gap_samples = int(round(args.gap * FS))
    air_wave, meta = build_air_waveform_selected(args.message, selected, air_repetitions=args.repetitions, gap_samples=gap_samples)
    air_duration = len(air_wave) / float(FS)
    rx_timeout = args.timeout if args.timeout is not None else (WARMUP_SECONDS + air_duration + TAIL_MARGIN_SECONDS)

    save_last = artifact_path(run_dir, "rx_fsk_capture.wav", args.save_last)
    json_out = artifact_path(run_dir, "rx_fsk_result.json", args.json_out)
    command_manifest(run_dir, sys.argv, {"profiles_spec": profiles_spec, "resolved_profiles": [p.name for p in selected], "air_duration_s": air_duration})

    receiver_cmd = [
        sys.executable, "-u", str(Path(__file__).with_name("receiver_fsk_final.py")),
        "--timeout", f"{rx_timeout:.1f}",
        "--profiles", profiles_spec,
        "--save-last", save_last,
        "--json-out", json_out,
    ]
    if args.device is not None:
        receiver_cmd += ["--device", str(args.device)]
    if args.verbose:
        receiver_cmd.append("--verbose")

    transmitter_cmd = [
        sys.executable, "-u", str(Path(__file__).with_name("transmitter_fsk_final.py")),
        "--repetitions", str(args.repetitions),
        "--gap", str(args.gap),
        "--profiles", profiles_spec,
        "--gain", str(args.gain),
        args.message,
    ]
    if args.device is not None:
        transmitter_cmd += ["--device", str(args.device)]

    log_path = artifact_path(run_dir, "commands.txt")
    write_text(log_path, "receiver: " + " ".join(receiver_cmd) + "\n" + "transmitter: " + " ".join(transmitter_cmd) + "\n")

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
