import subprocess
import sys
import time

RECEIVER = "receiver_v2.py"
TRANSMITTER = "transmitter_v2.py"
WARMUP_SECONDS = 1.5

def main():
    print(f"[auto_test] Starting receiver: {RECEIVER}")
    receiver = subprocess.Popen(
        [sys.executable, "-u", RECEIVER],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print(f"[auto_test] Waiting {WARMUP_SECONDS}s for receiver to begin listening...")
    time.sleep(WARMUP_SECONDS)

    print(f"[auto_test] Starting transmitter: {TRANSMITTER}")
    transmitter = subprocess.Popen(
        [sys.executable, "-u", TRANSMITTER],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    tx_rc = transmitter.wait()
    print(f"[auto_test] Transmitter exited with code {tx_rc}")

    rx_rc = receiver.wait()
    print(f"[auto_test] Receiver exited with code {rx_rc}")

    sys.exit(tx_rc or rx_rc)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[auto_test] Interrupted.")
        sys.exit(130)
