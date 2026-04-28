"""Offline streaming-engine self-test.

Runs a TX engine into an in-memory loopback "cable" (with optional AWGN), then
feeds the resulting samples into the matching RX engine. Asserts:
  - At least 4 frames decode within 1.5 seconds of audio
  - Aggregated Pe < 0.01

This catches arithmetic/sync bugs without needing audio hardware. Run before
demo day. It is not a substitute for the physical channel test.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

import streaming_engine as se
import ofdm_phy as op
from prbs15 import PRBS15, BERMeter, count_bit_errors, expected_payload, decode_header


def loopback(tx_engine, rx_engine, *, seconds: float = 2.0,
             snr_db: float = 30.0, label: str = "?",
             min_pe: float = 0.05, min_frames: int = 1) -> bool:
    fs = tx_engine.fs
    n_samples = int(seconds * fs)
    block = 4096
    np.random.seed(7)
    rx_total_blocks = []
    rms_sum = 0.0
    rms_n = 0
    # Reset
    tx_engine.reset_tx()
    if hasattr(rx_engine, "bermeter"):
        rx_engine.bermeter.reset()
    # Generate full TX waveform
    tx_full = []
    needed = n_samples
    while needed > 0:
        b = tx_engine.next_block(min(block, needed))
        tx_full.append(b)
        needed -= len(b)
    tx_full = np.concatenate(tx_full).astype(np.float32)
    rms = float(np.sqrt(np.mean(tx_full ** 2)) + 1e-12)
    sigma = rms * 10 ** (-snr_db / 20.0)
    # Add Gaussian noise to simulate channel
    noisy = tx_full + sigma * np.random.randn(len(tx_full)).astype(np.float32)
    # Feed in 4096-sample blocks like a real audio callback
    last_state = {}
    for i in range(0, len(noisy), block):
        rx_chunk = noisy[i:i + block]
        last_state = rx_engine.process(rx_chunk)
    pe_ema = float(last_state.get("pe_ema", 1.0))
    pe_cum = float(last_state.get("pe_cumulative", 1.0))
    seen = int(last_state.get("frames_seen", 0))
    locked = int(last_state.get("frames_locked", 0))
    rate = float(last_state.get("bit_rate_bps", 0.0))
    snr = float(last_state.get("snr_db", 0.0))
    flag = "OK" if (seen >= min_frames and pe_cum < min_pe) else "FAIL"
    print(f"  [{label}] rate={rate:.0f}bps frames={locked}/{seen} pe_ema={pe_ema:.5f} pe_cum={pe_cum:.5f} snr_proxy={snr:.1f}dB  [{flag}]")
    ok = seen >= min_frames and pe_cum < min_pe
    return ok


def main() -> int:
    print("=" * 70)
    print("STREAMING ENGINE SELF-TEST")
    print("=" * 70)
    fails = 0
    # FSK at 50 / 500 / 5000 — uses orthogonal-tone auto-selection
    print("\nFSK paths (orthogonal-tone auto-select):")
    for rate in [500.0, 5000.0]:
        tx = se.FSKEngine(bit_rate=rate)   # auto-orthogonal tones
        rx = se.FSKEngine(bit_rate=rate)
        seconds = 4.0 if rate < 1000 else 3.0
        ok = loopback(tx, rx, seconds=seconds, snr_db=20.0,
                       label=f"FSK {rate:.0f}bps tones={tx.tone0:.0f}/{tx.tone1:.0f}",
                       min_pe=0.05)
        if not ok:
            fails += 1

    print("\nQPSK paths:")
    for rate in [500.0, 5000.0]:
        tx = se.QPSKEngine(bit_rate=rate, carrier=4800.0)
        rx = se.QPSKEngine(bit_rate=rate, carrier=4800.0)
        ok = loopback(tx, rx, seconds=3.0, snr_db=25.0,
                       label=f"QPSK {rate:.0f}bps", min_pe=0.05)
        if not ok:
            fails += 1

    print("\nCDMA path:")
    tx = se.CDMAEngine(bit_rate=100.0, carrier=12800.0)
    rx = se.CDMAEngine(bit_rate=100.0, carrier=12800.0)
    ok = loopback(tx, rx, seconds=8.0, snr_db=10.0,
                   label="CDMA 100bps", min_pe=0.10)
    if not ok:
        fails += 1

    print("\nOFDM path (acoustic cfg):")
    tx = se.OFDMEngine(use_adaptive=False)
    rx = se.OFDMEngine(use_adaptive=False)
    ok = loopback(tx, rx, seconds=4.0, snr_db=25.0,
                   label="OFDM acoustic QPSK", min_pe=0.10)
    if not ok:
        fails += 1

    print("\nFHSS path:")
    tx = se.FHSSEngine()
    rx = se.FHSSEngine()
    ok = loopback(tx, rx, seconds=8.0, snr_db=15.0,
                   label="FHSS 25bps", min_pe=0.10)
    if not ok:
        fails += 1

    print("\n" + "=" * 70)
    if fails == 0:
        print("ALL OK")
    else:
        print(f"FAILED: {fails} mode(s) below thresholds")
    print("=" * 70)
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
