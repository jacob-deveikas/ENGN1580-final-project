"""Continuous streaming transmit/receive engines for the dashboard.

Every PHY exposes the same two methods:
  TXEngine.next_block(n_frames) -> np.float32 mono samples of length n_frames.
  RXEngine.process(samples) -> dict with keys: bit_errors, bits, locked,
                                                constellations, snr_per_bin_db,
                                                jammer_bins, mode, bit_rate_bps

The engines share frame-aligned PRBS-15 framing so a single sync slip costs
only one frame's worth of BER (Section 9.6 in the design doc). The visible
Pe meter is a 30-frame EMA, the headline number is the cumulative count.

All four PHYs (FSK, QPSK, CDMA, OFDM, FHSS) plug into the same engine, so the
dashboard can switch modes from a dropdown without touching the audio callback.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from prbs15 import (
    PRBS15, PRBS15_PERIOD, BERMeter, HEADER_BITS,
    encode_header, decode_header, count_bit_errors, expected_payload,
)

EPS = 1e-12
DEFAULT_FS = 48000


# ============================================================================
# 32-bit sync word with sharp autocorrelation. 32 bits = 16 QPSK symbols, so
# the HEADER (32 bits = 16 syms) and PAYLOAD (1024 bits = 512 syms) start on
# clean symbol boundaries with no half-symbol padding.
# ============================================================================
SYNC_WORD_BITS = np.array([
    1,1,1,1, 1,0,1,0, 1,1,0,0, 1,0,0,1,
    0,0,1,0, 1,1,0,1, 0,1,1,1, 0,0,0,1,
], dtype=np.uint8)
assert len(SYNC_WORD_BITS) == 32
SYNC_WORD_LEN = len(SYNC_WORD_BITS)
PAYLOAD_BITS_PER_FRAME = 1024


def _fft_correlate_real(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Sliding cross-correlation of x with h, returns 'valid' result of length
    len(x) - len(h) + 1. Uses FFT for speed.
    """
    x = np.asarray(x, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    n_full = len(x) + len(h) - 1
    n_fft = 1 << (n_full - 1).bit_length()
    X = np.fft.rfft(x, n_fft)
    H = np.fft.rfft(h[::-1], n_fft)
    y = np.fft.irfft(X * H, n_fft)[:n_full]
    return y[len(h) - 1:len(x)]


def _running_sum(x: np.ndarray, win: int) -> np.ndarray:
    """Length-(N-win+1) running sum of x with window `win`."""
    if len(x) < win:
        return np.zeros(0, dtype=np.float32)
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    return (c[win:] - c[:-win]).astype(np.float32)


def _build_frame_bits(prbs: PRBS15, n_payload_bits: int = PAYLOAD_BITS_PER_FRAME,
                      frame_no: int = 0) -> Tuple[np.ndarray, int]:
    """Compose [SYNC][HEADER][PAYLOAD] bits.

    Returns (bits, header_seed). The header_seed is what the receiver uses to
    re-seed its PRBS-15 and regenerate the expected payload.
    """
    header_seed = prbs.state & 0x7FFF
    payload = prbs.next_bits(n_payload_bits)
    header = encode_header(header_seed, frame_no)
    bits = np.concatenate([SYNC_WORD_BITS, header, payload]).astype(np.uint8)
    return bits, header_seed


# ============================================================================
# Frame-level BER bookkeeping shared across PHYs
# ============================================================================
@dataclass
class FrameStats:
    bit_errors: int = 0
    n_bits: int = 0
    locked: bool = False
    seed: int = 0
    frame_no: int = 0
    sync_score: float = 0.0


# ============================================================================
# Generic ring buffer for the audio callback to write into.
# ============================================================================
class RingBuffer:
    def __init__(self, n_samples: int):
        self.buf = np.zeros(int(n_samples), dtype=np.float32)
        self.cursor = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray) -> None:
        n = len(x)
        if n == 0:
            return
        with self.lock:
            if n >= len(self.buf):
                self.buf[:] = x[-len(self.buf):]
                self.cursor = 0
            else:
                end = self.cursor + n
                if end <= len(self.buf):
                    self.buf[self.cursor:end] = x
                else:
                    first = len(self.buf) - self.cursor
                    self.buf[self.cursor:] = x[:first]
                    self.buf[:n - first] = x[first:]
                self.cursor = (self.cursor + n) % len(self.buf)

    def snapshot(self, n: Optional[int] = None) -> np.ndarray:
        with self.lock:
            buf = self.buf.copy()
            cur = self.cursor
        ordered = np.concatenate([buf[cur:], buf[:cur]])
        if n is None or n >= len(ordered):
            return ordered
        return ordered[-n:].copy()


# ============================================================================
# FSK engine (continuous frame-aligned)
# ============================================================================
def fsk_orthogonal_tones(bit_rate: float, fs: int = DEFAULT_FS,
                          base_idx: int = 1, spacing_idx: int = 1) -> Tuple[float, float]:
    """Pick orthogonal FSK tones for the given bit rate.

    For matched-filter FSK demodulation to be clean, the integration window
    spb = round(fs / bit_rate) should contain integer periods of each tone
    AND the tone spacing should be a multiple of fs/spb (the local DFT bin
    spacing). This routine returns (tone0, tone1) satisfying both conditions
    while keeping both tones in the audible/passband range.
    """
    spb = max(2, int(round(fs / bit_rate)))
    bin_hz = fs / spb
    tone0 = base_idx * bin_hz
    tone1 = (base_idx + spacing_idx) * bin_hz
    # Snap into a useful audio range (200 Hz - 18 kHz)
    while tone1 > 18000.0 and base_idx > 1:
        base_idx -= 1
        tone0 = base_idx * bin_hz
        tone1 = (base_idx + spacing_idx) * bin_hz
    while tone0 < 200.0:
        base_idx += 1
        tone0 = base_idx * bin_hz
        tone1 = (base_idx + spacing_idx) * bin_hz
    return float(tone0), float(tone1)


class FSKEngine:
    def __init__(self, fs: int = DEFAULT_FS, bit_rate: float = 5000.0,
                 tone0: Optional[float] = None, tone1: Optional[float] = None,
                 band_low: Optional[float] = None, band_high: Optional[float] = None,
                 frame_size_bits: int = PAYLOAD_BITS_PER_FRAME):
        if tone0 is None or tone1 is None:
            tone0, tone1 = fsk_orthogonal_tones(bit_rate, fs)
        if band_low is None:
            band_low = max(200.0, min(tone0, tone1) - 800.0)
        if band_high is None:
            band_high = min(0.49 * fs, max(tone0, tone1) + 800.0)
        self.fs = fs
        self.bit_rate = float(bit_rate)
        self.tone0 = float(tone0)
        self.tone1 = float(tone1)
        self.band_low = float(band_low)
        self.band_high = float(band_high)
        self.spb = max(2, int(round(fs / bit_rate)))
        self.payload_bits = int(frame_size_bits)
        self.tx_prbs = PRBS15(seed=0x4A5B)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_phase = 0.0
        self.tx_frame_no = 0
        # RX
        self.rx_buffer = np.zeros(0, dtype=np.float32)
        self.bermeter = BERMeter(ema_window_frames=20)
        self.last_constellations: Dict[int, np.ndarray] = {}
        self.last_snr_db = 0.0
        self.lock = threading.Lock()

    def name(self) -> str:
        return f"FSK {self.bit_rate:.0f} bps ({self.tone0:.0f}/{self.tone1:.0f} Hz)"

    def bit_rate_bps(self) -> float:
        return self.fs / self.spb

    def _build_one_frame(self) -> np.ndarray:
        bits, _ = _build_frame_bits(self.tx_prbs, self.payload_bits, self.tx_frame_no)
        self.tx_frame_no += 1
        n = len(bits) * self.spb
        out = np.empty(n, dtype=np.float32)
        spb_inv = 1.0 / self.fs
        phase = self.tx_phase
        idx = 0
        for b in bits:
            f = self.tone1 if b else self.tone0
            for k in range(self.spb):
                out[idx] = math.sin(phase)
                phase += 2 * math.pi * f * spb_inv
                idx += 1
        # Wrap phase
        self.tx_phase = phase % (2 * math.pi)
        # Mild taper at frame boundaries to avoid clicks
        return out

    def next_block(self, n_frames: int) -> np.ndarray:
        while len(self.tx_buffer) < n_frames:
            f = self._build_one_frame()
            self.tx_buffer = np.concatenate([self.tx_buffer, f])
        out = self.tx_buffer[:n_frames]
        self.tx_buffer = self.tx_buffer[n_frames:]
        return (0.75 * out).astype(np.float32)

    def reset_tx(self, seed: int = 0x4A5B) -> None:
        self.tx_prbs.reset(seed)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_phase = 0.0
        self.tx_frame_no = 0

    def process(self, samples: np.ndarray) -> Dict:
        with self.lock:
            self.rx_buffer = np.concatenate([self.rx_buffer, samples.astype(np.float32)])
            frame_bits = SYNC_WORD_LEN + HEADER_BITS + self.payload_bits
            n_frame_samps = frame_bits * self.spb
            # We need at least one full frame plus a sync window of slack so
            # the sliding correlator can lock onto a frame whose sync starts
            # in the middle of the buffer.
            min_for_decode = n_frame_samps + SYNC_WORD_LEN * self.spb
            if len(self.rx_buffer) < min_for_decode:
                return self._dummy_result()
            self._try_decode_frames()
            # Trim to keep last 1 frame's worth + safety
            keep = max(2 * n_frame_samps, 8 * self.spb)
            if len(self.rx_buffer) > keep + n_frame_samps:
                self.rx_buffer = self.rx_buffer[-keep:]
            return {
                "mode": "fsk",
                "bit_rate_bps": self.bit_rate_bps(),
                "pe_ema": self.bermeter.pe_ema,
                "pe_cumulative": self.bermeter.pe_cumulative,
                "frames_locked": self.bermeter.frames_locked,
                "frames_seen": self.bermeter.frames_seen,
                "snr_db": float(self.last_snr_db),
                "constellations": self.last_constellations,
                "bermeter": self.bermeter,
            }

    def _dummy_result(self) -> Dict:
        return {
            "mode": "fsk",
            "bit_rate_bps": self.bit_rate_bps(),
            "pe_ema": self.bermeter.pe_ema,
            "pe_cumulative": self.bermeter.pe_cumulative,
            "frames_locked": self.bermeter.frames_locked,
            "frames_seen": self.bermeter.frames_seen,
            "snr_db": float(self.last_snr_db),
            "constellations": {},
            "bermeter": self.bermeter,
        }

    def _soft_continuous(self, x: np.ndarray) -> np.ndarray:
        """Sample-rate sliding (e1 - e0) matched filter.

        Returns array of length (len(x) - spb + 1). Each output sample is the
        windowed FSK soft metric over [t, t+spb). This is amplitude-invariant
        once we sign-threshold, and phase-invariant because we slide one
        sample at a time.
        """
        x = np.asarray(x, dtype=np.float32)
        if len(x) < self.spb + 4:
            return np.zeros(0, dtype=np.float32)
        t = np.arange(self.spb, dtype=np.float64) / self.fs
        c0 = np.cos(2 * math.pi * self.tone0 * t).astype(np.float32)
        s0 = np.sin(2 * math.pi * self.tone0 * t).astype(np.float32)
        c1 = np.cos(2 * math.pi * self.tone1 * t).astype(np.float32)
        s1 = np.sin(2 * math.pi * self.tone1 * t).astype(np.float32)
        # Sliding correlations. _fft_correlate_real returns valid output of
        # length len(x) - spb + 1.
        e0_c = _fft_correlate_real(x, c0)
        e0_s = _fft_correlate_real(x, s0)
        e1_c = _fft_correlate_real(x, c1)
        e1_s = _fft_correlate_real(x, s1)
        e0 = e0_c ** 2 + e0_s ** 2
        e1 = e1_c ** 2 + e1_s ** 2
        return (e1 - e0).astype(np.float32)

    def _try_decode_frames(self) -> None:
        frame_bits = SYNC_WORD_LEN + HEADER_BITS + self.payload_bits
        n_frame_samps = frame_bits * self.spb
        # Sample-rate continuous soft metric. soft[t] is (e1-e0) integrated
        # over the spb-sample window starting at t.
        soft = self._soft_continuous(self.rx_buffer)
        if len(soft) < (SYNC_WORD_LEN + HEADER_BITS + 4) * self.spb:
            return
        # Approach: for each phase 0..spb-1, sample soft at every spb-th sample
        # and look for the sync word in the resulting bit-rate stream. The
        # phase that gives the strongest sync correlation defines the bit
        # boundary. This is more robust than sample-rate sign correlation
        # because the soft signal is noisy at bit transitions.
        sync_signs = SYNC_WORD_BITS.astype(np.float32) * 2.0 - 1.0
        # For speed, only sweep coarse phases. spb is large at low rates.
        if self.spb >= 32:
            phases = list(range(0, self.spb, max(1, self.spb // 16)))
        else:
            phases = list(range(0, self.spb))
        best = (-1e18, 0, 0)  # (score, phase, peak_index_in_bits)
        sync_corrs_cache: Dict[int, np.ndarray] = {}
        for phase in phases:
            bit_softs = soft[phase::self.spb]
            n_bits = len(bit_softs)
            if n_bits < SYNC_WORD_LEN + HEADER_BITS + 4:
                continue
            # Sliding correlation against sync_signs in the bit stream.
            # Use raw soft (not sign) because magnitude weighting helps
            # rejection of transition-region noise.
            bs = bit_softs.astype(np.float32)
            sc = _fft_correlate_real(bs, sync_signs)
            if len(sc) == 0:
                continue
            # Normalize by local L2 of soft over SYNC_WORD_LEN bits
            local_e = _running_sum(bs ** 2, SYNC_WORD_LEN)
            denom = np.sqrt(np.maximum(local_e, 1e-12)) * math.sqrt(float(SYNC_WORD_LEN))
            min_len = min(len(sc), len(denom))
            sc_norm = sc[:min_len] / np.maximum(denom[:min_len], 1e-12)
            sync_corrs_cache[phase] = sc_norm
            if len(sc_norm) == 0:
                continue
            peak_idx = int(np.argmax(sc_norm))
            peak_val = float(sc_norm[peak_idx])
            if peak_val > best[0]:
                best = (peak_val, phase, peak_idx)
        peak_score, best_phase, _ = best
        if peak_score < 0.55:
            return
        sc_best = sync_corrs_cache[best_phase]
        # Find all peaks above threshold for this phase
        threshold = 0.55
        peaks: List[int] = []
        sep = frame_bits - 2
        i = 0
        while i < len(sc_best):
            if sc_best[i] >= threshold:
                end = min(i + max(8, frame_bits // 16), len(sc_best))
                p = i + int(np.argmax(sc_best[i:end]))
                peaks.append(p)
                i = p + sep
            else:
                i += 1
        if not peaks:
            return
        bit_softs_best = soft[best_phase::self.spb]
        decoded_to_sample = 0
        for sync_bit_idx in peaks:
            payload_bit_start = sync_bit_idx + SYNC_WORD_LEN
            need_bits = HEADER_BITS + self.payload_bits
            if payload_bit_start + need_bits > len(bit_softs_best):
                break
            payload_softs = bit_softs_best[payload_bit_start:payload_bit_start + need_bits]
            payload_signs = (payload_softs > 0).astype(np.uint8)
            header_bits = payload_signs[:HEADER_BITS]
            seed, frame_no = decode_header(header_bits)
            payload_bits_rx = payload_signs[HEADER_BITS:HEADER_BITS + self.payload_bits]
            expected = expected_payload(seed, self.payload_bits)
            errs = count_bit_errors(payload_bits_rx, expected)
            self.bermeter.update(errs, self.payload_bits, locked=True)
            self.last_snr_db = 10 * math.log10(max(float(sc_best[sync_bit_idx]), 1e-3)) + 20.0
            # Sample boundary in original rx buffer = best_phase + (payload_bit_start + need_bits) * spb
            sample_consumed = best_phase + (payload_bit_start + need_bits) * self.spb
            decoded_to_sample = max(decoded_to_sample, sample_consumed)
        if decoded_to_sample > 0 and decoded_to_sample < len(self.rx_buffer):
            self.rx_buffer = self.rx_buffer[decoded_to_sample:]


# ============================================================================
# QPSK engine (continuous, frame-aligned, with constellation output)
# ============================================================================
class QPSKEngine:
    QPSK_MAP = np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)], dtype=np.complex64) / math.sqrt(2.0)
    QPSK_BITS = ["00", "01", "11", "10"]

    def __init__(self, fs: int = DEFAULT_FS, bit_rate: float = 5000.0,
                 carrier: float = 4800.0,
                 frame_size_bits: int = PAYLOAD_BITS_PER_FRAME):
        self.fs = fs
        self.bit_rate = float(bit_rate)
        self.carrier = float(carrier)
        sym_rate = bit_rate / 2.0
        self.sps = max(4, int(round(fs / sym_rate)))
        self.payload_bits = int(frame_size_bits)
        self.tx_prbs = PRBS15(seed=0x4A5B)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_phase = 0.0
        self.tx_frame_no = 0
        self.rx_buffer = np.zeros(0, dtype=np.float32)
        self.rx_sample_idx = 0
        self.bermeter = BERMeter(ema_window_frames=20)
        self.last_constellation = np.zeros(0, dtype=np.complex64)
        self.last_snr_db = 0.0
        self.lock = threading.Lock()

    def name(self) -> str:
        return f"QPSK {self.bit_rate_bps():.0f} bps fc={self.carrier:.0f} Hz"

    def bit_rate_bps(self) -> float:
        return 2.0 * self.fs / self.sps

    def _bits_to_qpsk(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % 2:
            bits = np.concatenate([bits, [0]])
        b = bits.reshape(-1, 2)
        idx = b[:, 0] * 2 + b[:, 1]
        return self.QPSK_MAP[idx]

    def _build_one_frame(self) -> np.ndarray:
        bits, _ = _build_frame_bits(self.tx_prbs, self.payload_bits, self.tx_frame_no)
        self.tx_frame_no += 1
        syms = self._bits_to_qpsk(bits)
        base = np.repeat(syms, self.sps).astype(np.complex64)
        n = np.arange(len(base))
        phase_inc = 2 * math.pi * self.carrier / self.fs
        carrier_phase = self.tx_phase + phase_inc * n
        wave = np.real(base * np.exp(1j * carrier_phase)).astype(np.float32)
        self.tx_phase = (self.tx_phase + phase_inc * len(base)) % (2 * math.pi)
        return wave

    def next_block(self, n_frames: int) -> np.ndarray:
        while len(self.tx_buffer) < n_frames:
            f = self._build_one_frame()
            self.tx_buffer = np.concatenate([self.tx_buffer, f])
        out = self.tx_buffer[:n_frames]
        self.tx_buffer = self.tx_buffer[n_frames:]
        return (0.75 * out).astype(np.float32)

    def reset_tx(self, seed: int = 0x4A5B) -> None:
        self.tx_prbs.reset(seed)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_phase = 0.0
        self.tx_frame_no = 0

    def process(self, samples: np.ndarray) -> Dict:
        with self.lock:
            self.rx_buffer = np.concatenate([self.rx_buffer, samples.astype(np.float32)])
            frame_bits = SYNC_WORD_LEN + HEADER_BITS + self.payload_bits
            n_frame_samps = frame_bits * self.sps // 2
            # Need 1 frame + sync slack (sync sym count in samples)
            min_for_decode = n_frame_samps + SYNC_WORD_LEN * self.sps
            if len(self.rx_buffer) >= min_for_decode:
                self._try_decode()
            keep = max(2 * n_frame_samps, 4 * self.sps)
            if len(self.rx_buffer) > keep + n_frame_samps:
                self.rx_buffer = self.rx_buffer[-keep:]
            return {
                "mode": "qpsk",
                "bit_rate_bps": self.bit_rate_bps(),
                "pe_ema": self.bermeter.pe_ema,
                "pe_cumulative": self.bermeter.pe_cumulative,
                "frames_locked": self.bermeter.frames_locked,
                "frames_seen": self.bermeter.frames_seen,
                "snr_db": float(self.last_snr_db),
                "constellations": {0: self.last_constellation},
                "bermeter": self.bermeter,
            }

    def _try_decode(self) -> None:
        frame_bits = SYNC_WORD_LEN + HEADER_BITS + self.payload_bits
        assert frame_bits % 2 == 0, "frame_bits must be even for QPSK pairing"
        frame_syms = frame_bits // 2
        # Down-convert to baseband
        n = np.arange(len(self.rx_buffer))
        bb = 2.0 * self.rx_buffer * np.exp(-1j * 2 * math.pi * self.carrier * n / self.fs)
        if len(bb) < (frame_syms + 8) * self.sps:
            return
        sync_qpsk = self._bits_to_qpsk(SYNC_WORD_BITS)
        n_sync = len(sync_qpsk)   # 16 syms
        # Find best sample-level timing by sweeping offsets and checking
        # sync correlation at each. We pick the offset with the strongest
        # peak in the sync correlation.
        best_score, best_offset = -1.0, 0
        best_peak = 0
        sweep_step = max(1, self.sps // 16)
        for off in range(0, self.sps, sweep_step):
            usable_off = ((len(bb) - off) // self.sps) * self.sps
            if usable_off <= 0:
                continue
            ch = bb[off:off + usable_off].reshape(-1, self.sps)
            syms = ch.mean(axis=1)
            if len(syms) < frame_syms + 4:
                continue
            # Magnitude of sliding correlation against sync template.
            # We want peak |sum(syms_window * conj(sync))| relative to sync power.
            N = len(syms)
            L = n_sync
            n_fft = 1 << (N + L - 2).bit_length()
            A = np.fft.fft(syms, n_fft)
            B = np.fft.fft(np.conj(sync_qpsk[::-1]), n_fft)
            corr = np.fft.ifft(A * B)[:N + L - 1]
            valid = corr[L - 1:N]
            mag = np.abs(valid)
            if len(mag) == 0:
                continue
            peak_idx = int(np.argmax(mag))
            peak_val = float(mag[peak_idx])
            if peak_val > best_score:
                best_score = peak_val
                best_offset = off
                best_peak = peak_idx
        if best_score < 0.55 * n_sync:
            return
        # Re-derive symbols at the chosen offset
        usable_off = ((len(bb) - best_offset) // self.sps) * self.sps
        all_syms = bb[best_offset:best_offset + usable_off].reshape(-1, self.sps).mean(axis=1).astype(np.complex64)
        # Re-find all peaks at this offset
        N = len(all_syms)
        L = n_sync
        n_fft = 1 << (N + L - 2).bit_length()
        A = np.fft.fft(all_syms, n_fft)
        B = np.fft.fft(np.conj(sync_qpsk[::-1]), n_fft)
        corr = np.fft.ifft(A * B)[:N + L - 1]
        valid = corr[L - 1:N]
        mag = np.abs(valid)
        threshold = 0.55 * n_sync
        peaks: List[int] = []
        i = 0
        sep = frame_syms - 2
        while i < len(mag):
            if mag[i] > threshold:
                end = min(i + max(8, frame_syms // 4), len(mag))
                p = i + int(np.argmax(mag[i:end]))
                peaks.append(p)
                i = p + sep
            else:
                i += 1
        if not peaks:
            return
        decoded_to_sym = 0
        for p in peaks:
            payload_sym_start = p + n_sync
            n_payload_syms = (HEADER_BITS + self.payload_bits) // 2
            if payload_sym_start + n_payload_syms > len(all_syms):
                break
            sync_rx = all_syms[p:p + n_sync]
            # Phase + gain estimate using the entire sync sequence
            gain = np.vdot(sync_qpsk, sync_rx) / max(np.vdot(sync_qpsk, sync_qpsk).real, EPS)
            if abs(gain) < 1e-9:
                continue
            payload_rx = all_syms[payload_sym_start:payload_sym_start + n_payload_syms]
            payload_corr = payload_rx / gain
            # Hard demap to bits
            dist = np.abs(payload_corr.reshape(-1, 1) - self.QPSK_MAP.reshape(1, -1)) ** 2
            sym_idx = np.argmin(dist, axis=1)
            bits = np.zeros(2 * len(sym_idx), dtype=np.uint8)
            bits[0::2] = (sym_idx >> 1) & 1
            bits[1::2] = sym_idx & 1
            header_bits = bits[:HEADER_BITS]
            seed, frame_no = decode_header(header_bits)
            payload_bits_rx = bits[HEADER_BITS:HEADER_BITS + self.payload_bits]
            expected = expected_payload(seed, self.payload_bits)
            errs = count_bit_errors(payload_bits_rx, expected)
            self.bermeter.update(errs, self.payload_bits, locked=True)
            self.last_constellation = payload_corr[-400:].astype(np.complex64)
            err_norm = float(np.mean(np.abs(payload_corr - self.QPSK_MAP[sym_idx]) ** 2))
            sig_pwr = 1.0
            self.last_snr_db = 10 * math.log10(max(sig_pwr / max(err_norm, 1e-3), 1e-3))
            decoded_to_sym = payload_sym_start + n_payload_syms
        if decoded_to_sym > 0:
            sample_consumed = best_offset + decoded_to_sym * self.sps
            if sample_consumed < len(self.rx_buffer):
                self.rx_buffer = self.rx_buffer[sample_consumed:]


# ============================================================================
# CDMA engine — 64-chip BPSK at 100 bps, 12.8 kHz carrier
# ============================================================================
class CDMAEngine:
    def __init__(self, fs: int = DEFAULT_FS, bit_rate: float = 100.0,
                 carrier: float = 12800.0, n_chips: int = 64,
                 frame_size_bits: int = 256):
        self.fs = fs
        self.bit_rate = float(bit_rate)
        self.carrier = float(carrier)
        self.n_chips = int(n_chips)
        self.payload_bits = int(frame_size_bits)
        self.spb = max(2, int(round(fs / bit_rate)))
        self._chip_seq = self._make_mseq(n_chips)
        self.tx_prbs = PRBS15(seed=0x4A5B)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_frame_no = 0
        self.tx_phase = 0.0
        self.rx_buffer = np.zeros(0, dtype=np.float32)
        self.bermeter = BERMeter(ema_window_frames=8)
        self.last_constellation = np.zeros(0, dtype=np.complex64)
        self.last_snr_db = 0.0
        self.lock = threading.Lock()
        self._build_chip_template()

    def _make_mseq(self, length: int) -> np.ndarray:
        """Length-63 m-sequence (LFSR x^6+x+1), padded to 64."""
        if length != 64:
            rng = np.random.default_rng(12800)
            return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=length)
        state = 0b111111
        seq = []
        for _ in range(63):
            seq.append(1.0 if (state & 1) else -1.0)
            nb = ((state >> 0) ^ (state >> 1)) & 1
            state = (state >> 1) | (nb << 5)
        seq.append(seq[-1])  # pad to 64
        return np.array(seq, dtype=np.float32)

    def _build_chip_template(self) -> None:
        idx = np.floor(np.arange(self.spb) * self.n_chips / self.spb).astype(int)
        self.chips_per_bit = self._chip_seq[idx]
        t = np.arange(self.spb, dtype=np.float64) / self.fs
        self._cos = np.cos(2 * math.pi * self.carrier * t).astype(np.float32)
        self._sin = np.sin(2 * math.pi * self.carrier * t).astype(np.float32)
        self._tx_template_pos = (self.chips_per_bit * self._sin).astype(np.float32)
        self._matched_filter = self.chips_per_bit * self._sin

    def name(self) -> str:
        return f"CDMA-64 100 bps fc={self.carrier:.0f} Hz"

    def bit_rate_bps(self) -> float:
        return self.fs / self.spb

    def _build_one_frame(self) -> np.ndarray:
        bits, _ = _build_frame_bits(self.tx_prbs, self.payload_bits, self.tx_frame_no)
        self.tx_frame_no += 1
        n = len(bits) * self.spb
        out = np.empty(n, dtype=np.float32)
        for i, b in enumerate(bits):
            sign = 1.0 if b == 1 else -1.0
            out[i * self.spb:(i + 1) * self.spb] = sign * self._tx_template_pos
        return (0.65 * out).astype(np.float32)

    def next_block(self, n_frames: int) -> np.ndarray:
        while len(self.tx_buffer) < n_frames:
            f = self._build_one_frame()
            self.tx_buffer = np.concatenate([self.tx_buffer, f])
        out = self.tx_buffer[:n_frames]
        self.tx_buffer = self.tx_buffer[n_frames:]
        return out.astype(np.float32)

    def reset_tx(self, seed: int = 0x4A5B) -> None:
        self.tx_prbs.reset(seed)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_frame_no = 0

    def process(self, samples: np.ndarray) -> Dict:
        with self.lock:
            self.rx_buffer = np.concatenate([self.rx_buffer, samples.astype(np.float32)])
            frame_bits = SYNC_WORD_LEN + HEADER_BITS + self.payload_bits
            n_frame_samps = frame_bits * self.spb
            if len(self.rx_buffer) >= 2 * n_frame_samps:
                self._try_decode()
            keep = max(2 * n_frame_samps, 8 * self.spb)
            if len(self.rx_buffer) > keep + n_frame_samps:
                self.rx_buffer = self.rx_buffer[-keep:]
            return {
                "mode": "cdma",
                "bit_rate_bps": self.bit_rate_bps(),
                "pe_ema": self.bermeter.pe_ema,
                "pe_cumulative": self.bermeter.pe_cumulative,
                "frames_locked": self.bermeter.frames_locked,
                "frames_seen": self.bermeter.frames_seen,
                "snr_db": float(self.last_snr_db),
                "constellations": {0: self.last_constellation},
                "bermeter": self.bermeter,
            }

    def _try_decode(self) -> None:
        # Matched filter: correlate with one-bit template, sample every spb.
        x = self.rx_buffer
        if len(x) < 2 * self.spb:
            return
        n_full = len(x) + self.spb - 1
        n_fft = 1 << (n_full - 1).bit_length()
        X = np.fft.rfft(x.astype(np.float32), n_fft)
        H = np.fft.rfft(self._matched_filter[::-1], n_fft)
        corr = np.fft.irfft(X * H, n_fft)[:n_full]
        valid = corr[self.spb - 1:len(x)]
        # Take samples at every spb position; phase aligned by the largest |corr|^2 in first period
        if len(valid) < self.spb:
            return
        first_period = np.abs(valid[:self.spb]) ** 2
        phase = int(np.argmax(first_period))
        n_bits = (len(valid) - phase) // self.spb
        if n_bits < SYNC_WORD_LEN + HEADER_BITS + self.payload_bits + 4:
            return
        soft = valid[phase::self.spb][:n_bits]
        signs = (soft > 0).astype(np.uint8)
        # Find sync word
        sync_target = SYNC_WORD_BITS.astype(np.float32) * 2.0 - 1.0
        bit_signs = signs.astype(np.float32) * 2.0 - 1.0
        n_full2 = len(bit_signs) + len(sync_target) - 1
        n_fft2 = 1 << (n_full2 - 1).bit_length()
        S = np.fft.rfft(bit_signs, n_fft2)
        T = np.fft.rfft(sync_target[::-1], n_fft2)
        sync_corr = np.fft.irfft(S * T, n_fft2)[:n_full2]
        sync_valid = sync_corr[len(sync_target) - 1:len(bit_signs)]
        threshold = 0.55 * len(sync_target)
        if len(sync_valid) == 0:
            return
        peak_idx = int(np.argmax(sync_valid))
        if sync_valid[peak_idx] < threshold:
            return
        payload_start_bit = peak_idx + len(sync_target)
        if payload_start_bit + HEADER_BITS + self.payload_bits > len(signs):
            return
        header_bits = signs[payload_start_bit:payload_start_bit + HEADER_BITS]
        seed, frame_no = decode_header(header_bits)
        payload_rx = signs[payload_start_bit + HEADER_BITS:payload_start_bit + HEADER_BITS + self.payload_bits]
        expected = expected_payload(seed, self.payload_bits)
        errs = count_bit_errors(payload_rx, expected)
        self.bermeter.update(errs, self.payload_bits, locked=True)
        # Soft constellation: real part of normalized correlation
        # Use ~600 most recent symbols for the "constellation"
        consts = []
        for k in range(SYNC_WORD_LEN, min(n_bits, SYNC_WORD_LEN + HEADER_BITS + self.payload_bits)):
            consts.append(complex(float(soft[k]), 0.0))
        if consts:
            arr = np.asarray(consts, dtype=np.complex64)
            arr = arr / (np.median(np.abs(arr)) + EPS)
            self.last_constellation = arr[-400:]
        snr_proxy = float(np.mean(np.abs(soft) ** 2)) / max(float(np.var(soft - np.mean(soft))) + EPS, EPS)
        self.last_snr_db = 10.0 * math.log10(max(snr_proxy, 1e-3))
        sample_consumed = phase + (payload_start_bit + HEADER_BITS + self.payload_bits) * self.spb
        if sample_consumed < len(self.rx_buffer):
            self.rx_buffer = self.rx_buffer[sample_consumed:]


# ============================================================================
# OFDM engine — wildcard, highest data rate
# ============================================================================
import ofdm_phy as op


class OFDMEngine:
    def __init__(self, fs: int = DEFAULT_FS, cfg: Optional[op.OFDMConfig] = None,
                 loading_mode: Optional[int] = None,
                 max_loading: int = op.QAM64,
                 use_adaptive: bool = True,
                 frame_data_symbols: int = 32,
                 block_pilot_every: int = 16):
        self.fs = fs
        self.cfg = cfg or op.acoustic_config(fs=fs)
        # Default loading from cfg (BPSK for new acoustic, QPSK for wide/wired)
        if loading_mode is None:
            loading_mode = self.cfg.default_loading
        self.loading = op.BitLoadingMap.uniform(self.cfg, loading_mode)
        self.max_loading = int(max_loading)
        self.use_adaptive = bool(use_adaptive)
        self.frame_data_symbols = int(frame_data_symbols)
        self.block_pilot_every = int(block_pilot_every)
        self.tx_prbs = PRBS15(seed=0x4A5B)
        self.tx_frame_no = 0
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_preamble = op.chirp_preamble(self.cfg, duration_s=0.080)
        self.rx_buffer = np.zeros(0, dtype=np.float32)
        self.bermeter = BERMeter(ema_window_frames=8)
        self.last_constellations: Dict[int, np.ndarray] = {}
        self.last_snr_per_bin = np.zeros(self.cfg.n)
        self.last_snr_db = 0.0
        self.last_papr_db = 0.0
        self.lock = threading.Lock()
        # External hooks for backchannel feedback
        self.feedback_snr_db: Optional[np.ndarray] = None
        self.feedback_jammer_bins: List[int] = []
        self.frame_size_bits = self.loading.total_bits_per_symbol() * self.frame_data_symbols

    def name(self) -> str:
        thr = op.estimate_throughput(self.cfg, self.loading, self.block_pilot_every)
        return f"OFDM N={self.cfg.n} CP={self.cfg.cp} {thr['effective_bps']/1000:.1f} kbps"

    def bit_rate_bps(self) -> float:
        thr = op.estimate_throughput(self.cfg, self.loading, self.block_pilot_every)
        return float(thr["effective_bps"])

    def update_loading(self, snr_db: Optional[np.ndarray] = None,
                       jammer_bins: Optional[List[int]] = None) -> None:
        if not self.use_adaptive:
            return
        base = op.BitLoadingMap.uniform(self.cfg, op.QPSK)
        new_loading = base.with_overrides(snr_db=snr_db, jammer_bins=jammer_bins,
                                            max_loading=self.max_loading)
        self.loading = new_loading
        self.frame_size_bits = self.loading.total_bits_per_symbol() * self.frame_data_symbols

    def feed_backchannel(self, snr_db: np.ndarray, jammer_bins: List[int]) -> None:
        """Called by the dashboard when a feedback packet arrives."""
        self.feedback_snr_db = np.asarray(snr_db, dtype=np.float64)
        self.feedback_jammer_bins = list(jammer_bins)
        if self.use_adaptive:
            self.update_loading(snr_db=self.feedback_snr_db,
                                jammer_bins=self.feedback_jammer_bins)

    def _build_one_frame(self) -> np.ndarray:
        # FIXED-SEED frame design: every frame contains the SAME PRBS-15
        # payload starting from seed 0x4A5B. The RX has zero header
        # dependency — it always knows exactly what bits to compare.
        # If header decoding fails on the live channel, Pe is still
        # accurately measured.
        # The frame still has SYNC + HEADER for visual debugging, but
        # the BER comparator does not consume them.
        frame_no = self.tx_frame_no
        self.tx_frame_no += 1
        seed_fixed = 0x4A5B
        header = encode_header(seed_fixed, frame_no)
        payload_n = max(self.frame_size_bits - HEADER_BITS - SYNC_WORD_LEN, 64)
        # Re-seed PRBS for THIS frame so payload is deterministic
        prbs_fixed = PRBS15(seed=seed_fixed)
        payload = prbs_fixed.next_bits(payload_n)
        bits = np.concatenate([SYNC_WORD_BITS, header, payload]).astype(np.uint8)
        wave, used, offsets = op.modulate_frame(bits, self.cfg, self.loading,
                                                 n_data_symbols=self.frame_data_symbols,
                                                 block_pilot_every=self.block_pilot_every)
        full = np.concatenate([self.tx_preamble, wave]).astype(np.float32)
        return full

    def next_block(self, n_frames: int) -> np.ndarray:
        while len(self.tx_buffer) < n_frames:
            f = self._build_one_frame()
            self.tx_buffer = np.concatenate([self.tx_buffer, f])
        out = self.tx_buffer[:n_frames]
        self.tx_buffer = self.tx_buffer[n_frames:]
        return (0.85 * out).astype(np.float32)

    def reset_tx(self, seed: int = 0x4A5B) -> None:
        self.tx_prbs.reset(seed)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_frame_no = 0

    def _expected_bits_for_seed(self, seed: int) -> np.ndarray:
        """Reproduce expected payload. With FIXED-SEED frame design, the
        seed argument is ignored — every frame contains the same PRBS-15
        payload starting from 0x4A5B. RX is header-independent."""
        prbs = PRBS15(seed=0x4A5B)
        payload_n = max(self.frame_size_bits - HEADER_BITS - SYNC_WORD_LEN, 64)
        return prbs.next_bits(payload_n)

    def process(self, samples: np.ndarray) -> Dict:
        with self.lock:
            self.rx_buffer = np.concatenate([self.rx_buffer, samples.astype(np.float32)])
            frame_n_samples = len(self.tx_preamble) + (self.cfg.n + self.cfg.cp) * (self.frame_data_symbols + 4)
            if len(self.rx_buffer) >= 2 * frame_n_samples:
                self._try_decode()
            keep = max(2 * frame_n_samples, 4 * (self.cfg.n + self.cfg.cp))
            if len(self.rx_buffer) > keep + frame_n_samples:
                self.rx_buffer = self.rx_buffer[-keep:]
            return {
                "mode": "ofdm",
                "bit_rate_bps": self.bit_rate_bps(),
                "pe_ema": self.bermeter.pe_ema,
                "pe_cumulative": self.bermeter.pe_cumulative,
                "frames_locked": self.bermeter.frames_locked,
                "frames_seen": self.bermeter.frames_seen,
                "snr_db": float(self.last_snr_db),
                "snr_per_bin_db": self.last_snr_per_bin.copy(),
                "constellations": self.last_constellations,
                "papr_db": float(self.last_papr_db),
                "loading": self.loading.bits_per_bin.copy(),
                "bermeter": self.bermeter,
                "active_bins": self.cfg.active_bins(),
                "data_bins": self.cfg.data_bins(),
                "pilot_bins": self.cfg.pilot_bins(),
            }

    def _try_decode(self) -> None:
        # Find chirp preamble
        idx, score = op.detect_chirp(self.rx_buffer, self.tx_preamble)
        if idx is None or score < 0.20:
            return
        # OFDM block immediately follows the preamble
        block_start = idx + len(self.tx_preamble)
        # Plan to consume preamble + frame symbols
        max_syms = self.frame_data_symbols + (self.frame_data_symbols // self.block_pilot_every) + 1
        n_block_samps = (self.cfg.n + self.cfg.cp) * (max_syms + 1)
        if block_start + n_block_samps > len(self.rx_buffer):
            return
        rx = self.rx_buffer[block_start:block_start + n_block_samps]
        # Demodulate using a frame loading derived from feedback if available.
        loading_for_decode = self.loading
        # symbol offsets are evenly spaced (block pilot + data syms)
        sym_offsets = []
        cursor = self.cfg.n + self.cfg.cp
        for s in range(self.frame_data_symbols):
            if s > 0 and s % self.block_pilot_every == 0:
                cursor += (self.cfg.n + self.cfg.cp)
            sym_offsets.append(cursor)
            cursor += (self.cfg.n + self.cfg.cp)
        result = op.demodulate_frame(rx, self.cfg, loading_for_decode,
                                       data_symbol_offsets=sym_offsets,
                                       block_pilot_every=self.block_pilot_every)
        if not result.locked or len(result.bits) < SYNC_WORD_LEN + HEADER_BITS + 32:
            return
        bits = result.bits.astype(np.uint8)
        # Find sync word in the demodulated bit-stream (small window)
        if len(bits) < SYNC_WORD_LEN + HEADER_BITS + 16:
            return
        # Search for sync word with at most 6 errors in first 64 bits
        target = SYNC_WORD_BITS
        best_pos = -1
        best_match = -1
        for pos in range(0, min(64, len(bits) - len(target))):
            match = int(np.sum(bits[pos:pos + len(target)] == target))
            if match > best_match:
                best_match = match
                best_pos = pos
        if best_match < 0.7 * len(target):
            return
        header_start = best_pos + len(target)
        if header_start + HEADER_BITS > len(bits):
            return
        header_bits = bits[header_start:header_start + HEADER_BITS]
        seed, frame_no = decode_header(header_bits)
        # Reproduce expected payload from seed
        payload_rx = bits[header_start + HEADER_BITS:]
        expected = self._expected_bits_for_seed(seed)
        n_compare = min(len(payload_rx), len(expected))
        if n_compare < 64:
            return
        errs = count_bit_errors(payload_rx[:n_compare], expected[:n_compare])
        self.bermeter.update(errs, n_compare, locked=True)
        self.last_constellations = result.constellations
        self.last_snr_per_bin = result.snr_per_bin_db
        active = self.cfg.active_bins()
        if len(active):
            self.last_snr_db = float(np.median(result.snr_per_bin_db[active]))
        self.last_papr_db = result.papr_db
        # Pop processed
        consumed = block_start + n_block_samps
        if consumed < len(self.rx_buffer):
            self.rx_buffer = self.rx_buffer[consumed:]


# ============================================================================
# FHSS engine — bonus, frequency-hopping
# ============================================================================
import fhss_phy as fp


class FHSSEngine:
    def __init__(self, fs: int = DEFAULT_FS, cfg: Optional[fp.FHSSConfig] = None,
                 frame_size_bits: int = 96):
        self.fs = fs
        self.cfg = cfg or fp.FHSSConfig(fs=fs)
        self.payload_bits = int(frame_size_bits)
        self.tx_prbs = PRBS15(seed=0x4A5B)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_frame_no = 0
        self.preamble, _ = fp.make_preamble(self.cfg)
        self.rx_buffer = np.zeros(0, dtype=np.float32)
        self.bermeter = BERMeter(ema_window_frames=8)
        self.blacklist: set = set()
        self.last_snr_db = 0.0
        self.last_constellation = np.zeros(0, dtype=np.complex64)
        self.lock = threading.Lock()

    def name(self) -> str:
        return f"FHSS {self.cfg.n_slots} slots {self.cfg.bit_rate_bps():.0f} bps"

    def bit_rate_bps(self) -> float:
        return self.cfg.bit_rate_bps()

    def _build_one_frame(self) -> np.ndarray:
        seed = self.tx_prbs.state & 0x7FFF
        frame_no = self.tx_frame_no
        self.tx_frame_no += 1
        header = encode_header(seed, frame_no)
        payload = self.tx_prbs.next_bits(self.payload_bits)
        bits = np.concatenate([header, payload]).astype(np.uint8)
        wave, _ = fp.modulate_payload(self.cfg, bits, blacklist=self.blacklist)
        return np.concatenate([self.preamble, wave]).astype(np.float32)

    def next_block(self, n_frames: int) -> np.ndarray:
        while len(self.tx_buffer) < n_frames:
            f = self._build_one_frame()
            self.tx_buffer = np.concatenate([self.tx_buffer, f])
        out = self.tx_buffer[:n_frames]
        self.tx_buffer = self.tx_buffer[n_frames:]
        return (0.65 * out).astype(np.float32)

    def reset_tx(self, seed: int = 0x4A5B) -> None:
        self.tx_prbs.reset(seed)
        self.tx_buffer = np.zeros(0, dtype=np.float32)
        self.tx_frame_no = 0

    def update_blacklist(self, jammer_slots: set) -> None:
        self.blacklist = set(int(s) for s in jammer_slots)

    def process(self, samples: np.ndarray) -> Dict:
        with self.lock:
            self.rx_buffer = np.concatenate([self.rx_buffer, samples.astype(np.float32)])
            frame_n = len(self.preamble) + (HEADER_BITS + self.payload_bits) * self.cfg.samples_per_hop()
            if len(self.rx_buffer) >= 2 * frame_n:
                self._try_decode()
            keep = max(2 * frame_n, 4 * self.cfg.samples_per_hop())
            if len(self.rx_buffer) > keep + frame_n:
                self.rx_buffer = self.rx_buffer[-keep:]
            return {
                "mode": "fhss",
                "bit_rate_bps": self.bit_rate_bps(),
                "pe_ema": self.bermeter.pe_ema,
                "pe_cumulative": self.bermeter.pe_cumulative,
                "frames_locked": self.bermeter.frames_locked,
                "frames_seen": self.bermeter.frames_seen,
                "snr_db": float(self.last_snr_db),
                "blacklist": list(self.blacklist),
                "constellations": {0: self.last_constellation},
                "bermeter": self.bermeter,
            }

    def _try_decode(self) -> None:
        idx, score = fp.find_preamble(self.rx_buffer, self.cfg, threshold=0.30)
        if idx is None:
            return
        bit_start = idx + len(self.preamble)
        n_bits_total = HEADER_BITS + self.payload_bits
        n_samps = n_bits_total * self.cfg.samples_per_hop()
        if bit_start + n_samps > len(self.rx_buffer):
            return
        rx_bits, softs = fp.demodulate_payload(self.cfg, self.rx_buffer[bit_start:bit_start + n_samps],
                                                  n_bits_total, blacklist=self.blacklist)
        header_bits = rx_bits[:HEADER_BITS]
        seed, frame_no = decode_header(header_bits)
        payload_rx = rx_bits[HEADER_BITS:HEADER_BITS + self.payload_bits]
        expected = expected_payload(seed, self.payload_bits)
        errs = count_bit_errors(payload_rx, expected)
        self.bermeter.update(errs, self.payload_bits, locked=True)
        self.last_snr_db = float(score) * 30.0  # rough proxy
        self.last_constellation = (softs[-300:] / (np.median(np.abs(softs)) + EPS)).astype(np.complex64)
        consumed = bit_start + n_samps
        if consumed < len(self.rx_buffer):
            self.rx_buffer = self.rx_buffer[consumed:]


__all__ = [
    "FSKEngine", "QPSKEngine", "CDMAEngine", "OFDMEngine", "FHSSEngine",
    "PAYLOAD_BITS_PER_FRAME", "RingBuffer", "FrameStats",
    "SYNC_WORD_BITS", "SYNC_WORD_LEN",
]
