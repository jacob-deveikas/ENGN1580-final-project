from __future__ import annotations

import json
import math
import wave as wave_mod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, welch, csd, coherence
    HAVE_SCIPY = True
except Exception:
    butter = None
    sosfiltfilt = None
    welch = None
    csd = None
    coherence = None
    HAVE_SCIPY = False

FS = 44100
EPS = 1e-12
DEFAULT_GAIN = 0.75
PRE_GUARD_S = 0.080
POST_GUARD_S = 0.050
PRE_GUARD = int(round(PRE_GUARD_S * FS))
POST_GUARD = int(round(POST_GUARD_S * FS))
CHIRP_DURATION = 0.040
CHIRP_SAMPLES = int(round(CHIRP_DURATION * FS))
CHIRP_F0 = 1400.0
CHIRP_F1 = 3600.0
PREAMBLE_REPEAT = 3
PREAMBLE_THRESHOLD = 0.10

# Training allowed. It is not source/channel coding. It is synchronization and estimation.
SYNC_BITS = "111001011010000100101111011101000110101011001000"  # 48 fixed bits
QPSK_SYNC_SYMBOLS = 96
QPSK_RRC_ROLLOFF = 0.35
QPSK_RRC_SPAN = 8
QPSK_CARRIER_DEFAULT = 4800.0
QPSK_CARRIER_SEARCH_OFFSETS = [0, -100, 100, -200, 200, -400, 400, -700, 700, -1000, 1000]
QPSK_TIMING_SEARCH_SYMBOLS = 14

CDMA_RATE = 100.0
CDMA_CHIPS = 64
CDMA_CARRIER = 12800.0

MEASURE_DURATION_S = 2.0
MEASURE_F0 = 20.0
MEASURE_F1 = 20000.0
MEASURE_NPERSEG = 4096


@dataclass(frozen=True)
class FSKProfile:
    name: str
    tone0: float
    tone1: float
    band_low: float
    band_high: float


FSK_PROFILES = {
    "low": FSKProfile("low", 1900.0, 3100.0, 1200.0, 3800.0),
    "high": FSKProfile("high", 6600.0, 7800.0, 5600.0, 8800.0),
    "wide": FSKProfile("wide", 5200.0, 10800.0, 4200.0, 11800.0),
}


def list_profile_names() -> List[str]:
    return list(FSK_PROFILES.keys())


def parse_profiles(spec: str | None) -> List[FSKProfile]:
    if spec is None or str(spec).strip().lower() in {"", "auto", "diversity"}:
        return [FSK_PROFILES["low"], FSK_PROFILES["high"]]
    out = []
    for item in str(spec).split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in FSK_PROFILES:
            raise ValueError(f"unknown FSK profile {name!r}; choices={sorted(FSK_PROFILES)}")
        out.append(FSK_PROFILES[name])
    return out or [FSK_PROFILES["low"], FSK_PROFILES["high"]]


def _exp_sweep(n_samples: int, f0: float, f1: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / FS
    duration = max(n_samples / FS, EPS)
    ratio = max(f1 / max(f0, EPS), 1.000001)
    k = duration / math.log(ratio)
    phase = 2.0 * np.pi * f0 * k * (np.exp(t / k) - 1.0)
    y = np.sin(phase)
    ramp = min(len(y)//20, int(0.03*FS))
    if ramp > 1:
        win = np.hanning(2*ramp)
        y[:ramp] *= win[:ramp]
        y[-ramp:] *= win[ramp:]
    return (0.95 * y / (np.max(np.abs(y)) + EPS)).astype(np.float32)


def _lin_chirp(n_samples: int, f0: float, f1: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / FS
    duration = max(n_samples / FS, EPS)
    k = (f1 - f0) / duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    y = np.sin(phase) * np.hanning(n_samples)
    return (0.95 * y / (np.max(np.abs(y)) + EPS)).astype(np.float32)


UPCHIRP = _lin_chirp(CHIRP_SAMPLES, CHIRP_F0, CHIRP_F1)
DOWNCHIRP = _lin_chirp(CHIRP_SAMPLES, CHIRP_F1, CHIRP_F0)
PREAMBLE = np.concatenate([UPCHIRP, DOWNCHIRP] * PREAMBLE_REPEAT).astype(np.float32)
PREAMBLE_LEN = len(PREAMBLE)


def save_wav(path: str | Path, x: np.ndarray, fs: int = FS) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave_mod.open(str(p), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(pcm.tobytes())
    return str(p)


def load_wav(path: str | Path) -> np.ndarray:
    with wave_mod.open(str(path), "rb") as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        fs = f.getframerate()
        frames = f.getnframes()
        raw = f.readframes(frames)
    if fs != FS:
        raise ValueError(f"wav sample rate {fs} != expected {FS}")
    if sampwidth != 2:
        raise ValueError("only 16-bit PCM wav supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        x = x.reshape(-1, n_channels)[:, 0]
    return x.astype(np.float32)


def prbs_bits(n_bits: int, seed: int = 1580) -> str:
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 2, size=int(n_bits), dtype=np.uint8)
    return "".join("1" if int(v) else "0" for v in b)


def text_to_bits(text: str) -> str:
    return "".join(f"{b:08b}" for b in text.encode("utf-8"))


def bits_to_text(bits: str) -> str:
    usable = bits[: len(bits) - (len(bits) % 8)]
    data = bytes(int(usable[i:i+8], 2) for i in range(0, len(usable), 8))
    return data.decode("utf-8", errors="replace")


def robust_center_scale(x: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = float(np.std(x))
    if sigma < 1e-6:
        sigma = 1e-6
    return med, sigma


def clip_impulses(x: np.ndarray, clip_sigma: float = 5.0, blank_sigma: float = 9.0) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    if len(y) == 0:
        return y
    med, sigma = robust_center_scale(y)
    centered = y - med
    centered[np.abs(centered) > blank_sigma * sigma] = 0.0
    np.clip(centered, -clip_sigma * sigma, clip_sigma * sigma, out=centered)
    return (centered + med).astype(np.float32)


def bandpass(x: np.ndarray, low: float, high: float, order: int = 6) -> np.ndarray:
    if not HAVE_SCIPY or len(x) < 64:
        return np.asarray(x, dtype=np.float32)
    high = min(high, 0.49 * FS)
    low = max(low, 20.0)
    if high <= low:
        return np.asarray(x, dtype=np.float32)
    try:
        sos = butter(order, [low, high], fs=FS, btype="bandpass", output="sos")
        return sosfiltfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def _moving_sum(x: np.ndarray, win: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    return (c[win:] - c[:-win]).astype(np.float32)


def find_preamble(x: np.ndarray, threshold: float = PREAMBLE_THRESHOLD) -> Tuple[Optional[int], float]:
    sig = clip_impulses(np.asarray(x, dtype=np.float32))
    sig = bandpass(sig, 900.0, 4300.0)
    if len(sig) < PREAMBLE_LEN + 8:
        return None, 0.0
    tpl = PREAMBLE - np.mean(PREAMBLE)
    sig = sig - np.mean(sig)
    corr = np.correlate(sig, tpl, mode="valid")
    tpl_e = float(np.dot(tpl, tpl)) + EPS
    local = np.maximum(_moving_sum(sig * sig, PREAMBLE_LEN), 0.0)
    score = np.abs(corr) / (np.sqrt(local * tpl_e) + EPS)
    idx = int(np.argmax(score))
    best = float(score[idx])
    return (idx if best >= threshold else None), best


def assemble_packet(payload_wave: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.zeros(PRE_GUARD, dtype=np.float32),
        PREAMBLE,
        payload_wave.astype(np.float32),
        np.zeros(POST_GUARD, dtype=np.float32),
    ]).astype(np.float32)


def samples_per_bit(bit_rate: float) -> int:
    return max(2, int(round(FS / float(bit_rate))))


def actual_bit_rate(bit_rate: float) -> float:
    return FS / samples_per_bit(bit_rate)


def bit_errors(tx_bits: str, rx_bits: str) -> Tuple[int, int, float]:
    n = min(len(tx_bits), len(rx_bits))
    if n <= 0:
        return len(tx_bits), max(len(tx_bits), len(rx_bits), 1), 1.0
    errs = sum(a != b for a, b in zip(tx_bits[:n], rx_bits[:n])) + abs(len(tx_bits) - len(rx_bits))
    total = max(len(tx_bits), len(rx_bits))
    return errs, total, errs / max(total, 1)


# ---------------------- FSK graded, uncoded ----------------------

def build_fsk_payload(bits: str, bit_rate: float, profile: FSKProfile) -> Tuple[np.ndarray, Dict[str, float]]:
    spb = samples_per_bit(bit_rate)
    t = np.arange(spb, dtype=np.float64) / FS
    # Rectangular window preserves energy at high rates; mild cosine ramps avoid clicks at low rates.
    if spb >= 32:
        w = np.ones(spb)
        ramp = min(spb // 6, 8)
        if ramp > 1:
            r = np.sin(np.linspace(0, np.pi/2, ramp))**2
            w[:ramp] = r
            w[-ramp:] = r[::-1]
    else:
        w = np.ones(spb)
    s0 = np.sin(2*np.pi*profile.tone0*t) * w
    s1 = np.sin(2*np.pi*profile.tone1*t) * w
    out = np.empty(len(bits)*spb, dtype=np.float32)
    for i,b in enumerate(bits):
        out[i*spb:(i+1)*spb] = s1 if b == "1" else s0
    out = (0.75 * out / (np.max(np.abs(out)) + EPS)).astype(np.float32)
    return out, {"samples_per_bit": spb, "actual_rate": FS/spb, "tone0": profile.tone0, "tone1": profile.tone1}


def build_fsk_packet(bits: str, bit_rate: float, profiles: List[FSKProfile]) -> Tuple[np.ndarray, Dict[str, object]]:
    pieces: List[np.ndarray] = []
    metas = []
    gap = int(round(0.080*FS))
    for idx, p in enumerate(profiles):
        payload, meta = build_fsk_payload(SYNC_BITS + bits, bit_rate, p)
        pieces.append(assemble_packet(payload))
        metas.append({"profile": p.name, **meta})
        if idx != len(profiles)-1:
            pieces.append(np.zeros(gap, dtype=np.float32))
    wave = np.concatenate(pieces).astype(np.float32)
    return wave, {"mode": "fsk", "bit_rate_requested": bit_rate, "profiles": [p.name for p in profiles], "payload_bits": len(bits), "profile_meta": metas, "duration_s": len(wave)/FS}


def fsk_soft_metrics(x: np.ndarray, bit_rate: float, profile: FSKProfile, start: int, n_bits: int) -> Optional[np.ndarray]:
    spb = samples_per_bit(bit_rate)
    end = start + n_bits*spb
    if start < 0 or end > len(x):
        return None
    xx = bandpass(clip_impulses(x), profile.band_low, profile.band_high)
    segs = xx[start:end].reshape(n_bits, spb)
    segs = segs - np.mean(segs, axis=1, keepdims=True)
    t = np.arange(spb, dtype=np.float64) / FS
    refs = []
    for f in (profile.tone0, profile.tone1):
        refs.append((np.cos(2*np.pi*f*t).astype(np.float32), np.sin(2*np.pi*f*t).astype(np.float32)))
    c0,s0 = refs[0]
    c1,s1 = refs[1]
    e0 = (segs @ c0)**2 + (segs @ s0)**2
    e1 = (segs @ c1)**2 + (segs @ s1)**2
    return (e1 - e0).astype(np.float32)


def decode_fsk_capture(x: np.ndarray, bit_rate: float, n_payload_bits: int, profiles: List[FSKProfile], seed: int = 1580) -> Dict[str, object]:
    total_bits = len(SYNC_BITS) + int(n_payload_bits)
    best: Dict[str, object] = {"ok": False, "reason": "no_candidate", "score": -1.0, "mode": "fsk"}
    # Search all possible starts. Preamble helps, but blind sync rescues wired/audio offsets.
    pre_idx, pre_score = find_preamble(x, threshold=0.0)
    starts = []
    if pre_idx is not None:
        starts.append(pre_idx + PREAMBLE_LEN)
    spb = samples_per_bit(bit_rate)
    for rough in range(0, max(1, len(x) - total_bits*spb), max(1, spb*4)):
        starts.append(rough)
    tx_bits = prbs_bits(n_payload_bits, seed)
    sync_sign = np.array([1.0 if b == "1" else -1.0 for b in SYNC_BITS], dtype=np.float32)
    for profile in profiles:
        for start0 in starts:
            lo = max(0, int(start0) - 3*spb)
            hi = min(len(x) - total_bits*spb, int(start0) + 3*spb)
            if hi < lo:
                continue
            for s in range(lo, hi+1, max(1, spb//8)):
                soft = fsk_soft_metrics(x, bit_rate, profile, s, total_bits)
                if soft is None:
                    continue
                sync_soft = soft[:len(SYNC_BITS)]
                sync_score = float(np.dot(sync_soft, sync_sign) / (np.sum(np.abs(sync_soft))+EPS))
                if sync_score > float(best.get("score", -1.0)):
                    hard = "".join("1" if v > 0 else "0" for v in soft[len(SYNC_BITS):])[:n_payload_bits]
                    errs,total,pe = bit_errors(tx_bits, hard)
                    best = {"ok": pe < 0.01, "reason": "ok" if pe < 0.01 else "pe_too_high", "mode": "fsk", "profile": profile.name, "start": int(s), "score": sync_score, "preamble_score": pre_score, "received_bits": hard, "expected_bits": tx_bits, "bit_errors": errs, "n_bits": total, "pe": pe, "bit_rate_requested": bit_rate, "actual_bit_rate": actual_bit_rate(bit_rate)}
                    if best["ok"]:
                        return best
    return best


# ---------------------- QPSK graded, uncoded ----------------------

def rrc_taps(beta: float, span: int, sps: int) -> np.ndarray:
    n = np.arange(-span*sps, span*sps+1, dtype=np.float64)
    t = n / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4.0 / math.pi - 1.0)
        elif beta > 0 and abs(abs(4*beta*ti)-1.0) < 1e-9:
            taps[i] = beta/math.sqrt(2.0)*((1+2/math.pi)*math.sin(math.pi/(4*beta)) + (1-2/math.pi)*math.cos(math.pi/(4*beta)))
        else:
            num = math.sin(math.pi*ti*(1-beta)) + 4*beta*ti*math.cos(math.pi*ti*(1+beta))
            den = math.pi*ti*(1-(4*beta*ti)**2)
            taps[i] = num/den
    taps /= math.sqrt(np.sum(taps*taps) + EPS)
    return taps.astype(np.float32)


QPSK_MAP = {"00": (1+1j)/math.sqrt(2), "01": (-1+1j)/math.sqrt(2), "11": (-1-1j)/math.sqrt(2), "10": (1-1j)/math.sqrt(2)}
QPSK_CONST = np.array(list(QPSK_MAP.values()), dtype=np.complex64)
QPSK_BITS = ["00", "01", "11", "10"]


def bits_to_qpsk(bits: str) -> np.ndarray:
    if len(bits)%2:
        bits += "0"
    return np.array([QPSK_MAP[bits[i:i+2]] for i in range(0, len(bits), 2)], dtype=np.complex64)


def qpsk_to_bits(syms: np.ndarray) -> str:
    pts = np.asarray(syms, dtype=np.complex64).reshape(-1)
    dist = np.abs(pts[:,None] - QPSK_CONST[None,:])**2
    idx = np.argmin(dist, axis=1)
    return "".join(QPSK_BITS[int(i)] for i in idx)


def qpsk_sync_symbols() -> np.ndarray:
    return bits_to_qpsk(prbs_bits(2*QPSK_SYNC_SYMBOLS, seed=2604))


QPSK_SYNC = qpsk_sync_symbols()


def qpsk_params(bit_rate: float) -> Tuple[int, float, np.ndarray]:
    sym_rate = bit_rate / 2.0
    sps = max(3, int(round(FS / sym_rate)))
    actual_sym = FS / sps
    taps = rrc_taps(QPSK_RRC_ROLLOFF, QPSK_RRC_SPAN, sps)
    return sps, 2.0*actual_sym, taps


def build_qpsk_payload(bits: str, bit_rate: float, carrier: float) -> Tuple[np.ndarray, Dict[str, object]]:
    sps, actual_rate, taps = qpsk_params(bit_rate)
    syms = np.concatenate([np.zeros(8, dtype=np.complex64), QPSK_SYNC, bits_to_qpsk(bits), np.zeros(8, dtype=np.complex64)])
    up = np.zeros(len(syms)*sps, dtype=np.complex64)
    up[::sps] = syms
    shaped = np.convolve(up, taps.astype(np.complex64), mode="full")
    n = np.arange(len(shaped), dtype=np.float64)
    wave = np.real(shaped * np.exp(1j*2*np.pi*carrier*n/FS))
    wave = (0.75 * wave / (np.max(np.abs(wave))+EPS)).astype(np.float32)
    return wave, {"samples_per_symbol": sps, "actual_bit_rate": actual_rate, "carrier": carrier, "symbols": len(syms)}


def build_qpsk_packet(bits: str, bit_rate: float, carrier: float) -> Tuple[np.ndarray, Dict[str, object]]:
    payload, meta = build_qpsk_payload(bits, bit_rate, carrier)
    wave = assemble_packet(np.concatenate([np.zeros(int(0.01*FS), dtype=np.float32), payload, np.zeros(int(0.02*FS), dtype=np.float32)]))
    return wave, {"mode": "qpsk", "bit_rate_requested": bit_rate, "payload_bits": len(bits), "duration_s": len(wave)/FS, **meta}


def qpsk_downconvert(x: np.ndarray, carrier: float) -> np.ndarray:
    n = np.arange(len(x), dtype=np.float64)
    return np.asarray(x, dtype=np.float32) * np.exp(-1j*2*np.pi*carrier*n/FS)


def qpsk_decode_once(x: np.ndarray, bit_rate: float, n_payload_bits: int, carrier: float, seed: int = 1580) -> Dict[str, object]:
    pre_idx, pre_score = find_preamble(x, threshold=0.0)
    if pre_idx is None:
        return {"ok": False, "reason": "no_preamble", "mode": "qpsk", "carrier": carrier, "preamble_score": pre_score}
    start = pre_idx + PREAMBLE_LEN
    y = clip_impulses(x[start:])
    y = bandpass(y, max(100.0, carrier-2200.0), min(0.48*FS, carrier+2200.0))
    bb = qpsk_downconvert(y, carrier)
    sps, actual_rate, taps = qpsk_params(bit_rate)
    mf = np.convolve(bb, taps.astype(np.complex64), mode="full")
    delay = (len(taps)-1)//2
    nominal = int(0.01*FS) + 2*delay + 8*sps
    span = QPSK_TIMING_SEARCH_SYMBOLS*sps
    best_off, best_score, best_sync = None, -1.0, None
    max_idx = len(mf) - (QPSK_SYNC_SYMBOLS-1)*sps - 1
    for off in range(max(0, nominal-span), min(max_idx, nominal+span)+1):
        idx = off + np.arange(QPSK_SYNC_SYMBOLS)*sps
        rx_sync = mf[idx]
        score = float(abs(np.vdot(QPSK_SYNC, rx_sync)) / (np.sum(np.abs(rx_sync))+EPS))
        if score > best_score:
            best_score, best_off, best_sync = score, off, rx_sync
    if best_off is None:
        return {"ok": False, "reason": "bad_sync", "mode": "qpsk", "carrier": carrier, "preamble_score": pre_score, "sync_score": best_score}
    err = np.unwrap(np.angle(best_sync * np.conj(QPSK_SYNC)))
    n = np.arange(len(err), dtype=np.float64)
    slope, intercept = np.polyfit(n, err, 1) if len(err) > 1 else (0.0, float(np.mean(err)))
    total_syms = QPSK_SYNC_SYMBOLS + int(math.ceil(n_payload_bits/2))
    idx = best_off + np.arange(total_syms)*sps
    idx = idx[idx < len(mf)]
    rx = mf[idx]
    rot = np.exp(-1j*(slope*np.arange(len(rx)) + intercept))
    rx_corr = rx * rot
    gain = np.vdot(QPSK_SYNC, rx_corr[:QPSK_SYNC_SYMBOLS]) / (np.vdot(QPSK_SYNC, QPSK_SYNC)+EPS)
    if abs(gain) < 1e-6:
        gain = 1.0+0j
    rx_corr = rx_corr / gain
    data_syms = rx_corr[QPSK_SYNC_SYMBOLS:QPSK_SYNC_SYMBOLS+int(math.ceil(n_payload_bits/2))]
    rx_bits = qpsk_to_bits(data_syms)[:n_payload_bits]
    tx_bits = prbs_bits(n_payload_bits, seed)
    errs,total,pe = bit_errors(tx_bits, rx_bits)
    return {"ok": pe < 0.01, "reason": "ok" if pe < 0.01 else "pe_too_high", "mode": "qpsk", "carrier": carrier, "preamble_score": pre_score, "sync_score": best_score, "symbol_offset": int(best_off), "actual_bit_rate": actual_rate, "received_bits": rx_bits, "expected_bits": tx_bits, "bit_errors": errs, "n_bits": total, "pe": pe, "constellation": data_syms.astype(np.complex64), "sync_constellation": rx_corr[:QPSK_SYNC_SYMBOLS].astype(np.complex64)}


def decode_qpsk_capture(x: np.ndarray, bit_rate: float, n_payload_bits: int, carrier: float, seed: int = 1580, search: bool = True) -> Dict[str, object]:
    candidates = [carrier]
    if search:
        for d in QPSK_CARRIER_SEARCH_OFFSETS:
            f = carrier + d
            if 1800.0 <= f <= 15000.0 and f not in candidates:
                candidates.append(f)
    best = {"ok": False, "reason": "no_candidate", "mode": "qpsk", "sync_score": -1.0, "pe": 1.0}
    for fc in candidates:
        r = qpsk_decode_once(x, bit_rate, n_payload_bits, fc, seed=seed)
        score = float(r.get("sync_score", -1.0)) - 2.0*float(r.get("pe", 1.0))
        best_score = float(best.get("sync_score", -1.0)) - 2.0*float(best.get("pe", 1.0))
        if score > best_score:
            best = r
        if r.get("ok"):
            return r
    return best


# ---------------------- CDMA graded, uncoded ----------------------

def cdma_chip_sequence() -> np.ndarray:
    # Deterministic 64-chip +/-1 PN sequence.
    rng = np.random.default_rng(12800)
    chips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=CDMA_CHIPS)
    return chips.astype(np.float32)


CDMA_CHIP_SEQ = cdma_chip_sequence()


def cdma_symbol_wave(bit: str, rate: float = CDMA_RATE, carrier: float = CDMA_CARRIER) -> np.ndarray:
    spb = samples_per_bit(rate)
    idx = np.floor(np.arange(spb) * CDMA_CHIPS / spb).astype(int)
    chips = CDMA_CHIP_SEQ[idx]
    sign = 1.0 if bit == "1" else -1.0
    t = np.arange(spb, dtype=np.float64) / FS
    carrier_wave = np.sin(2*np.pi*carrier*t)
    w = np.ones(spb)
    ramp = min(spb//16, 8)
    if ramp > 1:
        r = np.sin(np.linspace(0, np.pi/2, ramp))**2
        w[:ramp] = r
        w[-ramp:] = r[::-1]
    return (sign * chips * carrier_wave * w).astype(np.float32)


def build_cdma_packet(bits: str, rate: float = CDMA_RATE, carrier: float = CDMA_CARRIER) -> Tuple[np.ndarray, Dict[str, object]]:
    waves = [cdma_symbol_wave(b, rate, carrier) for b in SYNC_BITS + bits]
    payload = 0.75*np.concatenate(waves).astype(np.float32)
    wave = assemble_packet(payload)
    return wave, {"mode": "cdma", "bit_rate_requested": rate, "actual_bit_rate": actual_bit_rate(rate), "payload_bits": len(bits), "carrier": carrier, "chips": CDMA_CHIPS, "duration_s": len(wave)/FS}


def decode_cdma_capture(x: np.ndarray, n_payload_bits: int, rate: float = CDMA_RATE, carrier: float = CDMA_CARRIER, seed: int = 1580) -> Dict[str, object]:
    pre_idx, pre_score = find_preamble(x, threshold=0.0)
    if pre_idx is None:
        return {"ok": False, "reason": "no_preamble", "mode": "cdma", "preamble_score": pre_score}
    start0 = pre_idx + PREAMBLE_LEN
    total_bits = len(SYNC_BITS) + n_payload_bits
    spb = samples_per_bit(rate)
    xx = bandpass(clip_impulses(x), max(100.0, carrier-5000), min(0.48*FS, carrier+5000))
    one = cdma_symbol_wave("1", rate, carrier)
    sync_sign = np.array([1 if b == "1" else -1 for b in SYNC_BITS], dtype=np.float32)
    best = {"score": -1.0}
    for start in range(max(0, start0-2*spb), min(len(xx)-total_bits*spb, start0+2*spb)+1, max(1, spb//16)):
        segs = xx[start:start+total_bits*spb].reshape(total_bits, spb)
        soft = segs @ one
        score = float(np.dot(soft[:len(SYNC_BITS)], sync_sign) / (np.sum(np.abs(soft[:len(SYNC_BITS)]))+EPS))
        if score > best.get("score", -1.0):
            hard = "".join("1" if v > 0 else "0" for v in soft[len(SYNC_BITS):])[:n_payload_bits]
            tx_bits = prbs_bits(n_payload_bits, seed)
            errs,total,pe = bit_errors(tx_bits, hard)
            best = {"ok": pe < 0.01, "reason": "ok" if pe < 0.01 else "pe_too_high", "mode": "cdma", "start": int(start), "score": score, "sync_score": score, "preamble_score": pre_score, "received_bits": hard, "expected_bits": tx_bits, "bit_errors": errs, "n_bits": total, "pe": pe, "actual_bit_rate": actual_bit_rate(rate), "carrier": carrier, "chips": CDMA_CHIPS}
            if best["ok"]:
                return best
    return best


# ---------------------- Measurement, adaptation, plots ----------------------

def build_measurement_waveform(duration_s: float = MEASURE_DURATION_S) -> Tuple[np.ndarray, Dict[str, object]]:
    sweep = _exp_sweep(int(round(duration_s*FS)), MEASURE_F0, MEASURE_F1)
    wave = np.concatenate([np.zeros(int(0.15*FS), dtype=np.float32), 0.65*sweep, np.zeros(int(0.15*FS), dtype=np.float32)]).astype(np.float32)
    return wave, {"mode": "sweep", "duration_s": len(wave)/FS, "sweep_duration_s": duration_s, "f0": MEASURE_F0, "f1": MEASURE_F1}


def estimate_frequency_response(recording: np.ndarray, duration_s: float = MEASURE_DURATION_S) -> Dict[str, object]:
    tx, meta = build_measurement_waveform(duration_s)
    x = np.asarray(recording, dtype=np.float32)
    if len(x) < len(tx):
        return {"ok": False, "reason": "too_short"}
    corr = np.correlate(x, tx, mode="valid")
    local = _moving_sum(x*x, len(tx))
    score = np.abs(corr)/(np.sqrt(local*(np.dot(tx,tx)+EPS))+EPS)
    idx = int(np.argmax(score))
    rx = x[idx:idx+len(tx)]
    if HAVE_SCIPY:
        nper = min(MEASURE_NPERSEG, len(tx))
        f, pxy = csd(tx, rx, fs=FS, nperseg=nper)
        _, pxx = welch(tx, fs=FS, nperseg=nper)
        h = pxy/(pxx+EPS)
        _, coh = coherence(tx, rx, fs=FS, nperseg=nper)
    else:
        nfft = 1
        while nfft < len(tx): nfft *= 2
        X = np.fft.rfft(tx, nfft)
        Y = np.fft.rfft(rx, nfft)
        h = Y/(X+EPS)
        f = np.fft.rfftfreq(nfft, 1/FS)
        coh = np.ones_like(f)
    mag_db = 20*np.log10(np.maximum(np.abs(h), 1e-9))
    rec = recommend_from_response(f, mag_db)
    return {"ok": True, "reason": "ok", "score": float(score[idx]), "freq": f, "mag_db": mag_db, "coherence": coh, "recommendation": rec, "meta": meta}


def _interp(f: np.ndarray, m: np.ndarray, hz: float) -> float:
    return float(np.interp(hz, f, m))


def recommend_from_response(freq: np.ndarray, mag_db: np.ndarray) -> Dict[str, object]:
    valid = (freq >= 1200) & (freq <= 15000)
    f = freq[valid]
    m = mag_db[valid]
    if len(f) < 3:
        return {"fsk_profile": "low", "qpsk_fc": QPSK_CARRIER_DEFAULT}
    width = max(5, len(m)//120)
    sm = np.convolve(m, np.ones(width)/width, mode="same")
    # Evaluate FSK profiles by weaker tone minus imbalance penalty.
    best_name, best_score = "low", -1e9
    evals = {}
    for name,p in FSK_PROFILES.items():
        if name == "wide":
            continue
        a = _interp(f, sm, p.tone0); b = _interp(f, sm, p.tone1)
        score = min(a,b) - 0.20*abs(a-b)
        evals[name] = {"tone0_db": a, "tone1_db": b, "imbalance_db": abs(a-b), "score": score}
        if score > best_score:
            best_name, best_score = name, score
    # QPSK carrier: strong and flat enough around bandwidth for 5 kbps wired/audio attempt.
    best_fc, best_q = QPSK_CARRIER_DEFAULT, -1e9
    for fc in np.arange(3000.0, 13000.0, 100.0):
        bw = 2500.0
        vals = [_interp(f, sm, fc-bw/2), _interp(f, sm, fc), _interp(f, sm, fc+bw/2)]
        flat = max(vals)-min(vals)
        score = min(vals) - 0.20*flat
        if score > best_q:
            best_q, best_fc = score, float(fc)
    return {"fsk_profile": best_name, "qpsk_fc": best_fc, "fsk_eval": evals, "notes": f"recommended FSK {best_name}; recommended QPSK carrier {best_fc:.0f} Hz"}


def ambient_scan(recording: np.ndarray) -> Dict[str, object]:
    x = np.asarray(recording, dtype=np.float32)
    if HAVE_SCIPY:
        f, pxx = welch(x, fs=FS, nperseg=min(4096, len(x)))
    else:
        X = np.fft.rfft(x*np.hanning(len(x)))
        f = np.fft.rfftfreq(len(x), 1/FS)
        pxx = np.abs(X)**2
    bands = {}
    for name,p in FSK_PROFILES.items():
        if name == "wide":
            continue
        mask = (f >= p.band_low) & (f <= p.band_high)
        bands[name] = float(10*np.log10(np.mean(pxx[mask])+EPS)) if np.any(mask) else 0.0
    selected = min(bands, key=bands.get)
    diff = abs(bands.get("low",0)-bands.get("high",0))
    if diff < 3.0:
        profiles = ["low", "high"]
        notes = "ambient roughly tied; use low,high diversity"
    else:
        profiles = [selected]
        notes = f"ambient favors {selected} by {diff:.1f} dB"
    return {"band_powers_db": bands, "selected_profiles": profiles, "notes": notes}


def plot_frequency_response(freq: np.ndarray, mag_db: np.ndarray, path: str | Path) -> str:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(freq, mag_db, lw=1.0)
    for name,p in FSK_PROFILES.items():
        if name != "wide":
            ax.axvspan(p.band_low, p.band_high, alpha=0.10, label=f"{name} FSK")
    ax.set_xlim(20, 20000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Speaker-Mic Magnitude Response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_ambient(bands: Dict[str, float], path: str | Path) -> str:
    import matplotlib.pyplot as plt
    keys = list(bands.keys())
    vals = [bands[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(keys, vals)
    ax.set_ylabel("Band power (dB, relative)")
    ax.set_title("Ambient Noise by FSK Band")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_constellation(points: np.ndarray, path: str | Path, title: str = "QPSK Constellation") -> str:
    import matplotlib.pyplot as plt
    pts = np.asarray(points, dtype=np.complex64).reshape(-1)
    fig, ax = plt.subplots(figsize=(6,6))
    if len(pts):
        ax.scatter(np.real(pts), np.imag(pts), s=14, alpha=0.70, label="rx symbols")
    ax.scatter(np.real(QPSK_CONST), np.imag(QPSK_CONST), marker="x", s=100, label="ideal")
    ax.axhline(0, lw=0.8)
    ax.axvline(0, lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    lim = max(1.8, float(np.max(np.abs(np.r_[np.real(pts), np.imag(pts), np.real(QPSK_CONST), np.imag(QPSK_CONST)]))) if len(pts) else 1.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_tx_waveform(wave: np.ndarray, path: str | Path, title: str = "Transmitter Waveform") -> str:
    import matplotlib.pyplot as plt
    x = np.asarray(wave, dtype=np.float32)
    t = np.arange(len(x))/FS
    show = min(len(x), int(0.25*FS))
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(t[:show], x[:show], lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_spectrum(wave: np.ndarray, path: str | Path, title: str = "Power Spectrum") -> str:
    import matplotlib.pyplot as plt
    x = np.asarray(wave, dtype=np.float32)
    if HAVE_SCIPY:
        f, pxx = welch(x, fs=FS, nperseg=min(4096, len(x)))
    else:
        X = np.fft.rfft(x*np.hanning(len(x)))
        f = np.fft.rfftfreq(len(x), 1/FS)
        pxx = np.abs(X)**2
    db = 10*np.log10(pxx+EPS)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(f, db, lw=1.0)
    ax.set_xlim(20, 20000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def save_frequency_csv(freq: np.ndarray, mag_db: np.ndarray, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, np.column_stack([freq, mag_db]), delimiter=",", header="frequency_hz,magnitude_db", comments="")
    return str(p)


def choose_carrier_from_file(path: str | None, fallback: float = QPSK_CARRIER_DEFAULT) -> float:
    if not path:
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rec = obj.get("recommendation", obj)
    return float(rec.get("qpsk_fc", fallback))


def choose_profiles_from_files(channel_file: str | None = None, ambient_file: str | None = None, explicit: str | None = None) -> str:
    if explicit and explicit.lower() not in {"auto", ""}:
        return explicit
    channel_prof = None
    ambient_prof = None
    if channel_file:
        try:
            with open(channel_file, "r", encoding="utf-8") as f:
                obj = json.load(f)
            rec = obj.get("recommendation", obj)
            channel_prof = rec.get("fsk_profile")
        except Exception:
            pass
    if ambient_file:
        try:
            with open(ambient_file, "r", encoding="utf-8") as f:
                obj = json.load(f)
            sel = obj.get("selected_profiles") or []
            if len(sel) == 1:
                ambient_prof = sel[0]
            elif len(sel) > 1:
                ambient_prof = ",".join(sel)
        except Exception:
            pass
    vals = [v for v in [channel_prof, ambient_prof] if v]
    if not vals:
        return "low,high"
    if len(vals) == 1:
        return vals[0]
    if vals[0] == vals[1]:
        return vals[0]
    return "low,high"
