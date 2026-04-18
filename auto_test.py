"""
auto_test.py -- drive the transmitter/receiver pair through a batch of
real over-the-air tests and report a pass/fail summary.

For each run:
  1. Start the receiver as a subprocess (it records for RECORD_DURATION seconds).
  2. Wait WARMUP_SECONDS so the receiver is actually listening.
  3. Start the transmitter as a subprocess to play the message.
  4. Collect both processes' stdout/stderr separately so the log is readable.
  5. Parse the receiver's final "Received message:" line and compare against
     the expected message to score pass/fail.

Usage:
    python auto_test.py                 # runs 5 HELLO tests (default)
    python auto_test.py 10              # runs 10 HELLO tests
    python auto_test.py 3 "TEST 123"    # runs 3 tests with a custom message

Note: the transmitter file is a module that calls transmit("HELLO") in its
__main__ block, so the default message is whatever that file sends. To send
a custom message, we override it via TX_MESSAGE_ENV (see below) -- this
requires a small change to transmitter_improved.py's __main__ block, which
is described in the README-style comment at the bottom of this file.
"""
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

RECEIVER    = "receiver_improved.py"
TRANSMITTER = "transmitter_improved.py"

WARMUP_SECONDS   = 1.5    # time for receiver to start listening before tx plays
COOLDOWN_SECONDS = 1.0    # gap between runs so late echoes don't bleed into next
DEFAULT_RUNS     = 5
DEFAULT_MESSAGE  = "HELLO"

# Regex to pull the final decoded message out of receiver stdout.
# Matches lines like:  Received message: 'HELLO'   or   Received message: None
_MSG_RE = re.compile(r"Received message:\s*(.+)$", re.MULTILINE)

@dataclass
class RunResult:
    index:    int
    expected: str
    decoded:  Optional[str]
    rx_log:   str
    tx_log:   str

    @property
    def passed(self) -> bool:
        return self.decoded == self.expected

def parse_received_message(rx_stdout: str) -> Optional[str]:
    """Pull the final 'Received message:' line out of receiver output."""
    matches = _MSG_RE.findall(rx_stdout)
    if not matches:
        return None
    raw = matches[-1].strip()
    if raw == "None":
        return None
    # Strip surrounding quotes from repr-style output: 'HELLO' -> HELLO
    if (raw.startswith("'") and raw.endswith("'")) or \
       (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1]
    return raw

def run_once(index: int, expected: str) -> RunResult:
    """Execute one receiver+transmitter pair and return the result."""
    print(f"\n{'=' * 60}")
    print(f"[auto_test] Run {index}: expecting {expected!r}")
    print('=' * 60)

    env = os.environ.copy()
    env["TX_MESSAGE"] = expected   # transmitter can read this if it wants to

    # Start receiver; capture its stdout/stderr separately from the transmitter.
    receiver = subprocess.Popen(
        [sys.executable, "-u", RECEIVER],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout for simpler parse
        text=True,
        env=env,
    )

    print(f"[auto_test] Receiver started (pid={receiver.pid}), warming up...")
    time.sleep(WARMUP_SECONDS)

    # Start transmitter.
    transmitter = subprocess.Popen(
        [sys.executable, "-u", TRANSMITTER],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    print(f"[auto_test] Transmitter started (pid={transmitter.pid})")

    # Wait for both to complete. Receiver has a fixed RECORD_DURATION, so it
    # will always exit on its own -- no need to terminate manually.
    tx_stdout, _ = transmitter.communicate()
    rx_stdout, _ = receiver.communicate()

    decoded = parse_received_message(rx_stdout)

    # Echo both logs so the user can still see what happened.
    print("\n--- transmitter output ---")
    print(tx_stdout.rstrip() or "(no output)")
    print("--- receiver output ---")
    print(rx_stdout.rstrip() or "(no output)")

    result = RunResult(index=index, expected=expected, decoded=decoded,
                       rx_log=rx_stdout, tx_log=tx_stdout)
    verdict = "PASS" if result.passed else "FAIL"
    print(f"\n[auto_test] Run {index} verdict: {verdict}  "
          f"(expected={expected!r}, decoded={decoded!r})")
    return result

def summarize(results: List[RunResult]) -> int:
    """Print a summary table and return a process exit code (0 = all pass)."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'#':<4}{'Verdict':<10}{'Expected':<14}{'Decoded':<30}")
    print("-" * 60)
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        decoded_disp = repr(r.decoded) if r.decoded is not None else "None"
        if len(decoded_disp) > 28:
            decoded_disp = decoded_disp[:25] + "..."
        print(f"{r.index:<4}{verdict:<10}{r.expected:<14}{decoded_disp:<30}")
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    print("-" * 60)
    print(f"Result: {passed}/{total} passed "
          f"({100 * passed / total:.0f}% success rate)")
    return 0 if passed == total else 1

def main(argv: List[str]) -> int:
    # Parse optional args: [n_runs] [message]
    n_runs  = int(argv[1]) if len(argv) > 1 else DEFAULT_RUNS
    message = argv[2] if len(argv) > 2 else DEFAULT_MESSAGE

    print(f"[auto_test] Running {n_runs} test(s), expected message: {message!r}")
    results: List[RunResult] = []
    for i in range(1, n_runs + 1):
        results.append(run_once(i, message))
        if i < n_runs:
            print(f"[auto_test] Cooling down {COOLDOWN_SECONDS}s before next run...")
            time.sleep(COOLDOWN_SECONDS)

    return summarize(results)

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print("\n[auto_test] Interrupted.")
        sys.exit(130)

# ---------------------------------------------------------------------------
# Optional: custom test messages per run
# ---------------------------------------------------------------------------
# If you want the transmitter to send whatever `TX_MESSAGE` the auto_test
# sets, change transmitter_improved.py's __main__ block from:
#
#     if __name__ == "__main__":
#         transmit("HELLO")
#
# to:
#
#     if __name__ == "__main__":
#         import os
#         transmit(os.environ.get("TX_MESSAGE", "HELLO"))
#
# Then `python auto_test.py 3 "TEST 123"` will send "TEST 123" each run.
# If you leave the transmitter unchanged, auto_test still works but will
# always send HELLO regardless of what you pass on the command line.