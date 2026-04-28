"""PRBS-15 LFSR with frame-aligned BER.

The reseed-per-frame design means a single missed frame costs only that frame's
worth of BER (1/window_size of the EMA). Without it a sync slip pegs Pe at 0.5
forever. This file is shared by the streaming TX, streaming RX, and dashboard.

The PRBS-15 polynomial is x^15 + x^14 + 1 (ITU-T O.151). Period = 32767.
"""

from __future__ import annotations

import struct
from collections import deque
from typing import Tuple

import numpy as np

PRBS15_PERIOD = 32767


class PRBS15:
    """ITU-T O.151 PRBS-15 generator. x^15 + x^14 + 1."""

    def __init__(self, seed: int = 0x4A5B):
        s = int(seed) & 0x7FFF
        if s == 0:
            s = 0x4A5B
        self.state = s

    def reset(self, seed: int) -> None:
        s = int(seed) & 0x7FFF
        if s == 0:
            s = 0x4A5B
        self.state = s

    def next_bit(self) -> int:
        nb = ((self.state >> 14) ^ (self.state >> 13)) & 1
        self.state = ((self.state << 1) | nb) & 0x7FFF
        return nb

    def next_bits(self, n: int) -> np.ndarray:
        out = np.empty(int(n), dtype=np.uint8)
        s = self.state
        for i in range(int(n)):
            nb = ((s >> 14) ^ (s >> 13)) & 1
            s = ((s << 1) | nb) & 0x7FFF
            out[i] = nb
        self.state = s
        return out


def make_payload(prbs: PRBS15, n_bits: int) -> Tuple[int, np.ndarray]:
    """Snapshot LFSR state, then emit `n_bits` PRBS bits.

    The header carries the snapshot; the receiver re-seeds a local PRBS-15
    from the header and regenerates the expected payload. The original PRBS
    state is advanced as a side-effect.
    """
    seed = prbs.state & 0x7FFF
    bits = prbs.next_bits(n_bits)
    return seed, bits


def expected_payload(seed: int, n_bits: int) -> np.ndarray:
    """Regenerate the bits the transmitter sent for this seed."""
    return PRBS15(seed=seed).next_bits(n_bits)


def encode_header(seed: int, frame_no: int) -> np.ndarray:
    """Build a 32-bit frame header. 16 bits seed, 16 bits frame counter.

    The frame counter rolls; it is for diagnostic only. The seed is the
    re-sync anchor.
    """
    s = int(seed) & 0xFFFF
    f = int(frame_no) & 0xFFFF
    pkt = struct.pack(">HH", s, f)
    bits = np.unpackbits(np.frombuffer(pkt, dtype=np.uint8))
    return bits.astype(np.uint8)


HEADER_BITS = 32


def decode_header(header_bits: np.ndarray) -> Tuple[int, int]:
    bb = np.asarray(header_bits[:HEADER_BITS], dtype=np.uint8)
    if len(bb) < HEADER_BITS:
        return 0, 0
    pkt = np.packbits(bb).tobytes()
    seed, frame_no = struct.unpack(">HH", pkt[:4])
    return int(seed), int(frame_no)


class BERMeter:
    """Frame-aligned running-average Pe meter.

    Use the EMA for the visible meter (responsive, ~window-frame settling).
    Use the cumulative count for the headline number ("we have measured
    N total bits with E errors, Pe=...").
    """

    def __init__(self, ema_window_frames: int = 30):
        self.alpha = 2.0 / (max(int(ema_window_frames), 1) + 1)
        self.pe_ema = 0.0
        self.total_bits = 0
        self.total_errors = 0
        self.frames_seen = 0
        self.frames_locked = 0
        self.history = deque(maxlen=2048)

    def reset(self) -> None:
        self.pe_ema = 0.0
        self.total_bits = 0
        self.total_errors = 0
        self.frames_seen = 0
        self.frames_locked = 0
        self.history.clear()

    def update(self, bit_errors: int, n_bits: int, locked: bool = True) -> None:
        if n_bits <= 0:
            return
        pe_inst = float(bit_errors) / float(n_bits)
        self.pe_ema = (1.0 - self.alpha) * self.pe_ema + self.alpha * pe_inst
        self.total_bits += int(n_bits)
        self.total_errors += int(bit_errors)
        self.frames_seen += 1
        if locked:
            self.frames_locked += 1
        self.history.append(self.pe_ema)

    @property
    def pe_cumulative(self) -> float:
        if self.total_bits == 0:
            return 1.0
        return self.total_errors / max(self.total_bits, 1)

    @property
    def lock_rate(self) -> float:
        if self.frames_seen == 0:
            return 0.0
        return self.frames_locked / self.frames_seen

    def summary(self) -> str:
        return (
            f"Pe(EMA)={self.pe_ema:.5f} "
            f"Pe(cum)={self.pe_cumulative:.5f} "
            f"errors={self.total_errors}/{self.total_bits} "
            f"frames={self.frames_locked}/{self.frames_seen}"
        )


def count_bit_errors(received: np.ndarray, expected: np.ndarray) -> int:
    n = min(len(received), len(expected))
    if n == 0:
        return 0
    a = np.asarray(received[:n], dtype=np.uint8)
    b = np.asarray(expected[:n], dtype=np.uint8)
    return int(np.sum(np.bitwise_xor(a, b)))


__all__ = [
    "PRBS15",
    "PRBS15_PERIOD",
    "BERMeter",
    "HEADER_BITS",
    "make_payload",
    "expected_payload",
    "encode_header",
    "decode_header",
    "count_bit_errors",
]
