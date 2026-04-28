"""Frequency-hopping spread spectrum (FHSS) BFSK.

Hops over 32 slots in [2.0, 10.0] kHz with 250 Hz spacing. Each hop carries
one BFSK bit (mark/space at +/-50 Hz around the slot center). The hop pattern
is generated from a shared SHA-256 seed so transmitter and receiver agree.

This is allowed under "no channel coding" because we are not adding parity
bits or redundancy. We are choosing where in the frequency band to put each
data bit. It is a transmission scheme, not an error-correction scheme.

Sync uses a known 16-symbol preamble at slot 0 (2.0 kHz). Acquisition is a
sliding correlation against the known preamble waveform.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

EPS = 1e-12


@dataclass
class FHSSConfig:
    fs: int = 48000
    n_slots: int = 32
    f0_hz: float = 2000.0
    slot_spacing_hz: float = 250.0
    hop_duration_s: float = 0.020
    bfsk_offset_hz: float = 50.0
    seed: bytes = b"chris-radio-april-30-2026"
    name: str = "fhss"

    def slot_centers(self) -> np.ndarray:
        return self.f0_hz + self.slot_spacing_hz * np.arange(self.n_slots)

    def samples_per_hop(self) -> int:
        return int(round(self.hop_duration_s * self.fs))

    def bit_rate_bps(self) -> float:
        # 1 BFSK bit per hop
        return 1.0 / self.hop_duration_s


def _hash_state(seed: bytes, n: int) -> int:
    h = hashlib.sha256(seed + n.to_bytes(8, "big")).digest()
    return int.from_bytes(h[:4], "big")


def hop_index(cfg: FHSSConfig, hop_n: int, blacklist: Optional[Set[int]] = None) -> int:
    """Return the slot for hop number `hop_n`, skipping blacklisted slots."""
    blacklist = blacklist or set()
    bl = {int(b) for b in blacklist if 0 <= int(b) < cfg.n_slots}
    n = int(hop_n)
    for _ in range(cfg.n_slots * 2):
        v = _hash_state(cfg.seed, n) % cfg.n_slots
        if v not in bl:
            return int(v)
        n += 1
    # Fallback if everything is blacklisted: use a deterministic non-blacklisted slot.
    for k in range(cfg.n_slots):
        if k not in bl:
            return k
    return 0


def make_hop_tone(cfg: FHSSConfig, slot: int, bit: int) -> np.ndarray:
    """Generate a single hop. Includes 5% taper to avoid clicks."""
    n = cfg.samples_per_hop()
    t = np.arange(n) / cfg.fs
    f_center = cfg.slot_centers()[slot]
    f = f_center + (cfg.bfsk_offset_hz if bit == 1 else -cfg.bfsk_offset_hz)
    y = np.sin(2 * math.pi * f * t)
    taper = max(8, n // 20)
    win = np.hanning(2 * taper)
    y[:taper] *= win[:taper]
    y[-taper:] *= win[taper:]
    return y.astype(np.float32)


def make_preamble(cfg: FHSSConfig) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Fixed preamble: 16 hops at slot 0 with alternating BFSK bits 1010..."""
    bits = [(0, k & 1 ^ 1) for k in range(16)]   # slot=0, bits = 1,0,1,0,...
    pieces = [make_hop_tone(cfg, slot, b) for slot, b in bits]
    return np.concatenate(pieces).astype(np.float32), bits


def modulate_payload(cfg: FHSSConfig, bits: np.ndarray,
                      blacklist: Optional[Set[int]] = None) -> Tuple[np.ndarray, List[int]]:
    """Modulate `bits` into FHSS waveform. Returns (samples, slot_sequence)."""
    pieces: List[np.ndarray] = []
    slots: List[int] = []
    for k, b in enumerate(bits.astype(int)):
        slot = hop_index(cfg, k, blacklist=blacklist)
        slots.append(slot)
        pieces.append(make_hop_tone(cfg, slot, int(b)))
    if not pieces:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(pieces).astype(np.float32), slots


def demodulate_hop(cfg: FHSSConfig, hop_samples: np.ndarray, slot: int) -> Tuple[int, float]:
    """Goertzel-ish energy comparison at the two BFSK tones for this slot."""
    n = len(hop_samples)
    if n < 8:
        return 0, 0.0
    t = np.arange(n) / cfg.fs
    f_center = cfg.slot_centers()[slot]
    f1 = f_center + cfg.bfsk_offset_hz
    f0 = f_center - cfg.bfsk_offset_hz
    c1 = np.cos(2 * math.pi * f1 * t).astype(np.float32)
    s1 = np.sin(2 * math.pi * f1 * t).astype(np.float32)
    c0 = np.cos(2 * math.pi * f0 * t).astype(np.float32)
    s0 = np.sin(2 * math.pi * f0 * t).astype(np.float32)
    e1 = (np.dot(hop_samples, c1) ** 2 + np.dot(hop_samples, s1) ** 2)
    e0 = (np.dot(hop_samples, c0) ** 2 + np.dot(hop_samples, s0) ** 2)
    soft = float(e1 - e0)
    return (1 if soft > 0 else 0), soft


def demodulate_payload(cfg: FHSSConfig, rx: np.ndarray, n_bits: int,
                        blacklist: Optional[Set[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate `n_bits` from `rx`. Assumes rx starts at the first hop boundary."""
    spb = cfg.samples_per_hop()
    bits = np.zeros(n_bits, dtype=np.uint8)
    softs = np.zeros(n_bits, dtype=np.float32)
    for k in range(n_bits):
        start = k * spb
        if start + spb > len(rx):
            break
        slot = hop_index(cfg, k, blacklist=blacklist)
        b, s = demodulate_hop(cfg, rx[start:start + spb], slot)
        bits[k] = b
        softs[k] = s
    return bits, softs


def detect_jammer_slots(rx_recent: np.ndarray, cfg: FHSSConfig, k_sigma: float = 5.0) -> Set[int]:
    """Run a Welch-style PSD and flag slots whose center has anomalous power."""
    if len(rx_recent) < 4096:
        return set()
    seg = rx_recent[-4096:] * np.hanning(4096)
    P = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(4096, 1 / cfg.fs)
    centers = cfg.slot_centers()
    bw = cfg.slot_spacing_hz
    slot_powers: List[float] = []
    for fc in centers:
        m = (freqs > fc - bw / 2) & (freqs < fc + bw / 2)
        slot_powers.append(float(P[m].sum()) if np.any(m) else 0.0)
    sp = np.array(slot_powers)
    med = float(np.median(sp) + 1e-12)
    mad = float(np.median(np.abs(sp - med)) + 1e-12)
    sigma = 1.4826 * mad
    bad = set(int(k) for k, v in enumerate(sp) if v > med + k_sigma * sigma)
    return bad


def find_preamble(rx: np.ndarray, cfg: FHSSConfig, threshold: float = 0.30) -> Tuple[Optional[int], float]:
    """Sliding correlation against the preamble waveform."""
    pre, _ = make_preamble(cfg)
    if len(rx) < len(pre) + 8:
        return None, 0.0
    rx_f = rx.astype(np.float32)
    pre_f = pre.astype(np.float32)
    n_full = len(rx_f) + len(pre_f) - 1
    n_fft = 1 << (n_full - 1).bit_length()
    R = np.fft.rfft(rx_f, n_fft)
    H = np.fft.rfft(pre_f[::-1], n_fft)
    y = np.fft.irfft(R * H, n_fft)[:n_full]
    valid = y[len(pre_f) - 1:len(rx_f)]
    e = np.cumsum(np.concatenate([[0], rx_f.astype(np.float64) ** 2]))
    if len(rx_f) - len(pre_f) + 1 <= 0:
        return None, 0.0
    local = e[len(pre_f):] - e[:len(rx_f) - len(pre_f) + 1]
    norm = np.sqrt(np.maximum(local * float(np.dot(pre_f, pre_f)), EPS))
    score = np.abs(valid[:len(norm)]) / norm
    if len(score) == 0:
        return None, 0.0
    idx = int(np.argmax(score))
    val = float(score[idx])
    if val < threshold:
        return None, val
    return idx, val


__all__ = [
    "FHSSConfig", "hop_index", "make_hop_tone", "make_preamble",
    "modulate_payload", "demodulate_payload", "demodulate_hop",
    "detect_jammer_slots", "find_preamble",
]
