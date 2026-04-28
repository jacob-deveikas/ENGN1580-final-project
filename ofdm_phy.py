"""Custom uncoded OFDM PHY.

This is V5's wildcard slot AND the "QPSK at multiple frequencies simultaneously"
bonus in one codebase.

Key parameters at FS=44100:
  N        = 1024              FFT size
  CP       = 128                cyclic prefix
  T_sym    = 1152 / 44100       = 26.1 ms  -> 38.3 OFDM sym/s
  Delta_f  = 44100 / 1024       = 43.07 Hz subcarrier spacing
  Pilots   = comb every 8 bins, BPSK +/- 1, +3 dB boost
  Acoustic active: bins 24-220   (1.03-9.48 kHz, 197 active)
  Wired    active: bins 8-460    (345-19.8 kHz,  453 active)
  Constellations: BPSK / QPSK / 16-QAM / 64-QAM / 256-QAM, gray-coded

There is NO channel coding here. There is NO source compression. Every bit on
the wire is a payload bit. Pilot SNR estimation drives the adaptive bit-loader,
which is independent and lives in `backchannel.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    butter = None
    sosfiltfilt = None

EPS = 1e-12

# ----------------------------- parameters ------------------------------------

DEFAULT_FS = 48000
DEFAULT_N = 1024
DEFAULT_CP = 128

# Bit loadings. 0 = subcarrier off, others = bits/symbol.
OFF, BPSK, QPSK, QAM16, QAM64, QAM256 = 0, 1, 2, 4, 6, 8
BIT_LOADINGS = (OFF, BPSK, QPSK, QAM16, QAM64, QAM256)

# Chow/Cioffi-style threshold lookup. dB column is rough Es/N0 needed for
# uncoded BER ~ 1e-2 with a small implementation margin.
SNR_THRESHOLDS_DB = {
    BPSK:  6.5,
    QPSK:  9.0,
    QAM16: 14.5,
    QAM64: 20.0,
    QAM256: 26.0,
}


@dataclass
class OFDMConfig:
    fs: int = DEFAULT_FS
    n: int = DEFAULT_N
    cp: int = DEFAULT_CP
    f_low_hz: float = 1000.0
    f_high_hz: float = 9500.0
    pilot_stride: int = 8
    pilot_boost_db: float = 3.0
    default_loading: int = QPSK
    papr_clip_ratio: float = 1.5      # 1.0 = no clip, lower = more aggressive
    pilot_seed: int = 0xC0DE
    name: str = "ofdm-acoustic"

    @property
    def t_sym(self) -> float:
        return (self.n + self.cp) / self.fs

    @property
    def delta_f(self) -> float:
        return self.fs / self.n

    def first_bin(self) -> int:
        return max(1, int(math.ceil(self.f_low_hz / self.delta_f)))

    def last_bin(self) -> int:
        return min(self.n // 2 - 1, int(math.floor(self.f_high_hz / self.delta_f)))

    def active_bins(self) -> np.ndarray:
        return np.arange(self.first_bin(), self.last_bin() + 1, dtype=np.int32)

    def pilot_bins(self) -> np.ndarray:
        active = self.active_bins()
        if len(active) == 0:
            return active
        first = active[0]
        last = active[-1]
        return np.arange(first, last + 1, self.pilot_stride, dtype=np.int32)

    def data_bins(self) -> np.ndarray:
        a = set(self.active_bins().tolist())
        p = set(self.pilot_bins().tolist())
        return np.array(sorted(a - p), dtype=np.int32)


def acoustic_config(fs: int = DEFAULT_FS) -> OFDMConfig:
    """Narrow-band BPSK loading for noisy/reverberant rooms. Concentrates
    transmit power in the cleanest band (1.5-5 kHz) where MacBook speaker
    response is flat AND ambient noise is moderate. ~80 subcarriers x BPSK
    = ~3 kbps net. Lower rate but reliable Pe<0.01 in real lab environments.
    For high-SNR demos, switch to wider band / higher loading.
    """
    return OFDMConfig(fs=fs, name="ofdm-acoustic", f_low_hz=1500.0, f_high_hz=5000.0,
                      default_loading=BPSK, papr_clip_ratio=1.5)


def acoustic_wide_config(fs: int = DEFAULT_FS) -> OFDMConfig:
    """Original wide-band acoustic config. Use only when SNR is high
    (very close laptops, quiet room)."""
    return OFDMConfig(fs=fs, name="ofdm-acoustic-wide", f_low_hz=1000.0, f_high_hz=9500.0,
                      default_loading=QPSK, papr_clip_ratio=1.5)


def wired_config(fs: int = DEFAULT_FS) -> OFDMConfig:
    return OFDMConfig(fs=fs, name="ofdm-wired", f_low_hz=350.0, f_high_hz=19500.0,
                      default_loading=QAM16, papr_clip_ratio=1.6)


# ----------------------------- gray QAM helpers ------------------------------

def _gray_pam(bits_per_dim: int) -> np.ndarray:
    """Gray-coded PAM levels for one dimension.

    Returns an array of length 2**bits_per_dim mapping integer index (Gray ord)
    to a real PAM amplitude in {+/-1, +/-3, ...} normalized to unit average
    energy.
    """
    m = 1 << bits_per_dim
    # Gray code permutation
    levels = np.arange(m)
    gray = levels ^ (levels >> 1)
    inv = np.zeros(m, dtype=int)
    for i, g in enumerate(gray):
        inv[g] = i
    raw = np.array([2 * inv[i] - (m - 1) for i in range(m)], dtype=np.float64)
    norm = math.sqrt(np.mean(raw ** 2))
    return raw / norm


def _qam_constellation(bits_per_sym: int) -> Tuple[np.ndarray, int]:
    """Return constellation array of size 2**bits_per_sym and bits_per_sym.

    For square QAM (4, 16, 64, 256) the I and Q legs use bits_per_sym//2 bits.
    BPSK uses pure real {+1,-1}.
    """
    if bits_per_sym == 1:
        return np.array([1.0 + 0j, -1.0 + 0j], dtype=np.complex64), 1
    half = bits_per_sym // 2
    pam = _gray_pam(half)
    pts = np.empty((1 << bits_per_sym,), dtype=np.complex64)
    for k in range(1 << bits_per_sym):
        i_part = k >> half
        q_part = k & ((1 << half) - 1)
        pts[k] = pam[i_part] + 1j * pam[q_part]
    return pts, bits_per_sym


_QAM_CACHE: Dict[int, Tuple[np.ndarray, int]] = {}


def qam_constellation(bits_per_sym: int) -> np.ndarray:
    if bits_per_sym not in _QAM_CACHE:
        _QAM_CACHE[bits_per_sym] = _qam_constellation(bits_per_sym)
    return _QAM_CACHE[bits_per_sym][0]


def bits_to_qam(bits: np.ndarray, bits_per_sym: int) -> np.ndarray:
    """Pack `bits_per_sym` bits at a time into a QAM symbol."""
    pts = qam_constellation(bits_per_sym)
    bb = np.asarray(bits, dtype=np.uint8)
    if len(bb) % bits_per_sym:
        pad = bits_per_sym - len(bb) % bits_per_sym
        bb = np.concatenate([bb, np.zeros(pad, dtype=np.uint8)])
    bb = bb.reshape(-1, bits_per_sym)
    idx = np.zeros(bb.shape[0], dtype=np.int32)
    for k in range(bits_per_sym):
        idx = (idx << 1) | bb[:, k].astype(np.int32)
    return pts[idx]


def qam_demap(syms: np.ndarray, bits_per_sym: int) -> np.ndarray:
    """Hard nearest-neighbor demap. Returns bits."""
    pts = qam_constellation(bits_per_sym)
    s = np.asarray(syms, dtype=np.complex64).reshape(-1, 1)
    p = pts.reshape(1, -1)
    dist = np.abs(s - p) ** 2
    idx = np.argmin(dist, axis=1).astype(np.uint32)
    out = np.zeros((len(syms), bits_per_sym), dtype=np.uint8)
    for k in range(bits_per_sym):
        out[:, k] = (idx >> (bits_per_sym - 1 - k)) & 1
    return out.reshape(-1)


# ----------------------------- pilot symbol ----------------------------------

def pilot_symbol(cfg: OFDMConfig) -> np.ndarray:
    """Generate the BPSK pilot pattern in the frequency domain (length N)."""
    rng = np.random.default_rng(cfg.pilot_seed)
    out = np.zeros(cfg.n, dtype=np.complex64)
    pilots = cfg.pilot_bins()
    boost = 10 ** (cfg.pilot_boost_db / 20.0)
    out[pilots] = boost * rng.choice([1.0 + 0j, -1.0 + 0j], size=len(pilots))
    return out


def block_pilot_symbol(cfg: OFDMConfig) -> np.ndarray:
    """Full-bandwidth BPSK block pilot for channel re-estimation."""
    rng = np.random.default_rng(cfg.pilot_seed ^ 0xA5)
    out = np.zeros(cfg.n, dtype=np.complex64)
    active = cfg.active_bins()
    boost = 10 ** (cfg.pilot_boost_db / 20.0)
    out[active] = boost * rng.choice([1.0 + 0j, -1.0 + 0j], size=len(active))
    return out


# ----------------------------- modulator -------------------------------------

@dataclass
class BitLoadingMap:
    """Per-bin bits/sym. Length cfg.n. Pilots/DC/edge bins are 0."""
    cfg: OFDMConfig
    bits_per_bin: np.ndarray

    @classmethod
    def uniform(cls, cfg: OFDMConfig, bits_per_sym: int) -> "BitLoadingMap":
        m = np.zeros(cfg.n, dtype=np.int32)
        m[cfg.data_bins()] = bits_per_sym
        return cls(cfg=cfg, bits_per_bin=m)

    def total_bits_per_symbol(self) -> int:
        return int(np.sum(self.bits_per_bin))

    def bit_rate_bps(self) -> float:
        return self.total_bits_per_symbol() / self.cfg.t_sym

    def with_overrides(self, snr_db: np.ndarray | None = None,
                       jammer_bins: List[int] | None = None,
                       max_loading: int = QAM64) -> "BitLoadingMap":
        m = self.bits_per_bin.copy()
        active = self.cfg.data_bins()
        if snr_db is not None:
            for k in active:
                if k >= len(snr_db):
                    continue
                s = float(snr_db[k])
                if s < SNR_THRESHOLDS_DB[BPSK]:
                    m[k] = OFF
                elif s < SNR_THRESHOLDS_DB[QPSK]:
                    m[k] = BPSK
                elif s < SNR_THRESHOLDS_DB[QAM16]:
                    m[k] = QPSK
                elif s < SNR_THRESHOLDS_DB[QAM64]:
                    m[k] = QAM16
                elif s < SNR_THRESHOLDS_DB[QAM256]:
                    m[k] = QAM64
                else:
                    m[k] = min(QAM256, max_loading)
        if jammer_bins:
            for k in jammer_bins:
                if 0 <= k < len(m):
                    m[k] = OFF
        # Limit to max_loading
        m = np.where(m > max_loading, max_loading, m)
        return BitLoadingMap(cfg=self.cfg, bits_per_bin=m)


def _papr_clip(time_sym: np.ndarray, ratio: float, fs: int,
               band_low: float, band_high: float) -> np.ndarray:
    """1-iteration iterative clipping + filter. Trims peaks to ratio*RMS,
    then bandpass-filters back into the active band.

    The dashboard reports the PAPR in dB so you can see this live.
    """
    if ratio is None or ratio >= 100:
        return time_sym
    rms = float(np.sqrt(np.mean(time_sym ** 2)) + EPS)
    thr = ratio * rms
    clipped = np.clip(time_sym, -thr, thr)
    if not HAVE_SCIPY:
        return clipped.astype(np.float32)
    try:
        lo = max(20.0, band_low - 100.0)
        hi = min(0.49 * fs, band_high + 100.0)
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        return sosfiltfilt(sos, clipped).astype(np.float32)
    except Exception:
        return clipped.astype(np.float32)


def modulate_one_symbol(payload_bits: np.ndarray, cfg: OFDMConfig,
                         loading: BitLoadingMap, include_pilots: bool = True,
                         block_pilot: bool = False) -> Tuple[np.ndarray, int]:
    """Build one OFDM time-domain symbol with CP. Returns (samples, bits_used)."""
    X = np.zeros(cfg.n, dtype=np.complex64)
    bits_used = 0
    pilots = cfg.pilot_bins()
    if include_pilots:
        ps = pilot_symbol(cfg)
        X[pilots] = ps[pilots]
    if block_pilot:
        # Override entire active band with the deterministic block pilot.
        bp = block_pilot_symbol(cfg)
        active = cfg.active_bins()
        X[active] = bp[active]
        # block-pilot symbols carry no payload bits.
        x_time = np.fft.ifft(X).real * cfg.n
        x_time = x_time / (np.sqrt(np.mean(x_time ** 2)) + EPS)
        x_with_cp = np.concatenate([x_time[-cfg.cp:], x_time]).astype(np.float32)
        return x_with_cp, 0
    # Data bins
    for k in cfg.data_bins():
        bps = int(loading.bits_per_bin[k])
        if bps == 0:
            continue
        if bits_used + bps > len(payload_bits):
            # not enough payload, pad with zeros
            chunk = np.concatenate([payload_bits[bits_used:], np.zeros(bps - (len(payload_bits) - bits_used), dtype=np.uint8)])
            sym = bits_to_qam(chunk, bps)[0]
            bits_used = len(payload_bits)
        else:
            chunk = payload_bits[bits_used:bits_used + bps]
            sym = bits_to_qam(chunk, bps)[0]
            bits_used += bps
        X[k] = sym
    # Hermitian symmetry not needed because we iFFT-take-real and use only k <= N/2-1.
    # But to keep the time-domain real we must mirror conjugate.
    for k in range(1, cfg.n // 2):
        X[cfg.n - k] = np.conj(X[k])
    X[0] = 0.0
    X[cfg.n // 2] = 0.0
    x_time = np.fft.ifft(X).real * cfg.n
    rms = float(np.sqrt(np.mean(x_time ** 2)) + EPS)
    x_time = x_time / rms
    x_time = _papr_clip(x_time, cfg.papr_clip_ratio, cfg.fs, cfg.f_low_hz, cfg.f_high_hz)
    x_with_cp = np.concatenate([x_time[-cfg.cp:], x_time]).astype(np.float32)
    return x_with_cp, bits_used


def modulate_frame(payload_bits: np.ndarray, cfg: OFDMConfig,
                    loading: BitLoadingMap, n_data_symbols: int,
                    block_pilot_every: int = 16) -> Tuple[np.ndarray, int, List[int]]:
    """Build a sequence of OFDM symbols. The first symbol is a block pilot
    so the receiver gets a clean channel estimate immediately.

    Returns (waveform, payload_bits_consumed, list_of_data_symbol_offsets).
    """
    pieces: List[np.ndarray] = []
    bits_used = 0
    data_symbol_offsets: List[int] = []
    # Start with a block pilot
    bp, _ = modulate_one_symbol(np.zeros(0, dtype=np.uint8), cfg, loading,
                                  include_pilots=False, block_pilot=True)
    pieces.append(bp)
    cursor = len(bp)
    for s in range(n_data_symbols):
        if s > 0 and (s % block_pilot_every) == 0:
            bp, _ = modulate_one_symbol(np.zeros(0, dtype=np.uint8), cfg, loading,
                                          include_pilots=False, block_pilot=True)
            pieces.append(bp)
            cursor += len(bp)
        x, used = modulate_one_symbol(payload_bits[bits_used:], cfg, loading,
                                        include_pilots=True, block_pilot=False)
        data_symbol_offsets.append(cursor)
        pieces.append(x)
        cursor += len(x)
        bits_used += used
        if bits_used >= len(payload_bits):
            break
    out = np.concatenate(pieces).astype(np.float32)
    peak = float(np.max(np.abs(out)) + EPS)
    out = (0.85 * out / peak).astype(np.float32)
    return out, bits_used, data_symbol_offsets


# ----------------------------- demodulator -----------------------------------

@dataclass
class DemodResult:
    bits: np.ndarray
    constellations: Dict[int, np.ndarray]      # bin -> array of symbols (post-eq)
    snr_per_bin_db: np.ndarray
    H_est: np.ndarray
    n_data_symbols: int
    papr_db: float
    locked: bool


def _channel_estimate_block(rx_block_freq: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """LS channel estimate at a block-pilot symbol. Returns H of length N."""
    bp = block_pilot_symbol(cfg)
    H = np.ones(cfg.n, dtype=np.complex64)
    active = cfg.active_bins()
    H[active] = rx_block_freq[active] / (bp[active] + EPS)
    # Smooth H across active band by short DFT-domain smoothing
    h_active = H[active]
    if len(h_active) > 9:
        # 5-tap moving average
        kernel = np.ones(5) / 5.0
        smoothed = np.convolve(h_active, kernel, mode="same")
        H[active] = smoothed.astype(np.complex64)
    return H


def _channel_update_pilots(rx_freq: np.ndarray, H_prev: np.ndarray,
                             cfg: OFDMConfig, pilot_phase_track: bool = True) -> Tuple[np.ndarray, float]:
    """Mid-frame pilot tracking: returns updated H and detected CFO phase slope."""
    pilots = cfg.pilot_bins()
    ps = pilot_symbol(cfg)
    H = H_prev.copy()
    if len(pilots) == 0:
        return H, 0.0
    H_pil = rx_freq[pilots] / (ps[pilots] + EPS)
    # Common phase rotation = mean argument of (H_pil / H_prev[pilots])
    rel = H_pil * np.conj(H_prev[pilots]) / (np.abs(H_prev[pilots]) ** 2 + EPS)
    phi = float(np.angle(np.mean(rel))) if pilot_phase_track else 0.0
    H *= np.exp(1j * phi)
    return H, phi


def demodulate_frame(rx: np.ndarray, cfg: OFDMConfig, loading: BitLoadingMap,
                      data_symbol_offsets: List[int],
                      block_pilot_every: int = 16) -> DemodResult:
    """Demodulate a frame of OFDM symbols. `rx` starts at the block pilot."""
    rx = np.asarray(rx, dtype=np.float32)
    sym_len = cfg.n + cfg.cp
    if len(rx) < sym_len:
        return DemodResult(bits=np.zeros(0, dtype=np.uint8), constellations={},
                            snr_per_bin_db=np.zeros(cfg.n), H_est=np.ones(cfg.n, dtype=np.complex64),
                            n_data_symbols=0, papr_db=0.0, locked=False)
    # 1) Initial block pilot at offset 0
    Y0 = np.fft.fft(rx[cfg.cp:cfg.cp + cfg.n])
    H = _channel_estimate_block(Y0, cfg)

    bits_out: List[np.ndarray] = []
    constellations: Dict[int, List[complex]] = {int(k): [] for k in cfg.data_bins()}
    noise_var_per_bin = np.full(cfg.n, 1e-3, dtype=np.float64)
    locked_count = 0

    n_data = 0
    cursor = sym_len  # first data sym after block pilot
    sym_idx = 0
    block_pilot_count = 1
    while cursor + sym_len <= len(rx) and n_data < len(data_symbol_offsets):
        # Refresh channel estimate at block-pilot symbols
        if sym_idx > 0 and (sym_idx % block_pilot_every) == 0:
            Yb = np.fft.fft(rx[cursor + cfg.cp:cursor + cfg.cp + cfg.n])
            H = _channel_estimate_block(Yb, cfg)
            cursor += sym_len
            block_pilot_count += 1
            continue
        # Data symbol
        Y = np.fft.fft(rx[cursor + cfg.cp:cursor + cfg.cp + cfg.n])
        H, _phi = _channel_update_pilots(Y, H, cfg)
        # 1-tap MMSE eq
        Heq = np.conj(H) / (np.abs(H) ** 2 + 1e-3)
        Xhat = Y * Heq
        # Per-bin EVM-based noise estimate (use distance to nearest constellation point)
        for k in cfg.data_bins():
            bps = int(loading.bits_per_bin[k])
            if bps == 0:
                continue
            constellations[int(k)].append(complex(Xhat[k]))
            pts = qam_constellation(bps)
            d2 = np.min(np.abs(Xhat[k] - pts) ** 2)
            # EWMA noise estimate
            noise_var_per_bin[k] = 0.9 * noise_var_per_bin[k] + 0.1 * float(d2)
        # Hard-decode bits this symbol
        sym_bits: List[np.ndarray] = []
        for k in cfg.data_bins():
            bps = int(loading.bits_per_bin[k])
            if bps == 0:
                continue
            sym_bits.append(qam_demap(np.array([Xhat[k]]), bps))
        if sym_bits:
            bits_out.append(np.concatenate(sym_bits))
        n_data += 1
        sym_idx += 1
        cursor += sym_len
        locked_count += 1
    # Per-bin SNR estimate from EVM
    sig_pwr = 1.0  # constellations are unit-mean-energy
    snr_db = np.zeros(cfg.n)
    for k in cfg.active_bins():
        n_var = max(noise_var_per_bin[k], 1e-6)
        snr_db[k] = 10.0 * math.log10(sig_pwr / n_var)
    # PAPR of the rx
    papr_db = 20.0 * math.log10((np.max(np.abs(rx)) + EPS) /
                                 (np.sqrt(np.mean(rx ** 2)) + EPS))
    consts_arrays = {k: np.asarray(v, dtype=np.complex64) for k, v in constellations.items() if v}
    bits = np.concatenate(bits_out) if bits_out else np.zeros(0, dtype=np.uint8)
    return DemodResult(bits=bits, constellations=consts_arrays,
                        snr_per_bin_db=snr_db, H_est=H,
                        n_data_symbols=n_data, papr_db=float(papr_db), locked=locked_count > 0)


# ----------------------------- preamble (chirp) ------------------------------

def chirp_preamble(cfg: OFDMConfig, duration_s: float = 0.080) -> np.ndarray:
    """Linear chirp 1->15 kHz, hann-windowed. Constant-envelope so it is
    PAPR-clip immune. Used for coarse symbol alignment.
    """
    n = int(round(duration_s * cfg.fs))
    t = np.arange(n) / cfg.fs
    f0, f1 = max(800.0, cfg.f_low_hz), min(15000.0, cfg.f_high_hz - 100.0)
    k = (f1 - f0) / max(duration_s, 1e-9)
    phase = 2 * math.pi * (f0 * t + 0.5 * k * t * t)
    y = np.sin(phase) * np.hanning(n)
    return (0.85 * y / (np.max(np.abs(y)) + EPS)).astype(np.float32)


def detect_chirp(rx: np.ndarray, ref: np.ndarray) -> Tuple[Optional[int], float]:
    """Matched-filter chirp detection. Returns (peak_index_in_rx, score)."""
    if len(rx) < len(ref) + 8:
        return None, 0.0
    n_full = len(rx) + len(ref) - 1
    n_fft = 1 << (n_full - 1).bit_length()
    R = np.fft.rfft(rx, n_fft)
    H = np.fft.rfft(ref[::-1], n_fft)
    y = np.fft.irfft(R * H, n_fft)[:n_full]
    valid = y[len(ref) - 1:len(rx)]
    # Normalize by local energy so the score is amplitude-invariant.
    e = np.cumsum(np.concatenate([[0], rx.astype(np.float64) ** 2]))
    local = e[len(ref):] - e[:len(rx) - len(ref) + 1]
    norm = np.sqrt(np.maximum(local * float(np.dot(ref, ref)), 1e-9))
    score = np.abs(valid[:len(norm)]) / norm
    if len(score) == 0:
        return None, 0.0
    idx = int(np.argmax(score))
    return idx, float(score[idx])


# ----------------------------- public API ------------------------------------

def estimate_throughput(cfg: OFDMConfig, loading: BitLoadingMap,
                         block_pilot_every: int = 16) -> Dict[str, float]:
    """Return realistic throughput accounting for block-pilot overhead."""
    payload_bits_per_sym = float(loading.total_bits_per_symbol())
    overhead = 1.0 / float(block_pilot_every + 1)   # 1 pilot per N+1 symbols
    eff = (1.0 - overhead) * payload_bits_per_sym / cfg.t_sym
    return {
        "raw_bits_per_symbol": payload_bits_per_sym,
        "raw_bps": payload_bits_per_sym / cfg.t_sym,
        "effective_bps": eff,
        "t_sym_ms": cfg.t_sym * 1000.0,
        "delta_f_hz": cfg.delta_f,
        "n_active": int(len(cfg.active_bins())),
        "n_data": int(len(cfg.data_bins())),
        "n_pilot": int(len(cfg.pilot_bins())),
    }


__all__ = [
    "OFDMConfig", "BitLoadingMap", "DemodResult",
    "OFF", "BPSK", "QPSK", "QAM16", "QAM64", "QAM256",
    "SNR_THRESHOLDS_DB",
    "acoustic_config", "wired_config",
    "modulate_one_symbol", "modulate_frame",
    "demodulate_frame",
    "chirp_preamble", "detect_chirp",
    "qam_constellation", "bits_to_qam", "qam_demap",
    "estimate_throughput",
    "pilot_symbol", "block_pilot_symbol",
]
