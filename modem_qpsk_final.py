
import json
import math
import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
    HAVE_SCIPY = True
except Exception:
    butter = None
    sosfiltfilt = None
    HAVE_SCIPY = False

FS = 44100
EPS = 1e-12

# Keep the preamble in the speaker-mic friendly low band for acquisition.
PRE_GUARD_S = 0.050
POST_GUARD_S = 0.030
PRE_GUARD_SAMPLES = int(round(PRE_GUARD_S * FS))
POST_GUARD_SAMPLES = int(round(POST_GUARD_S * FS))
CHIRP_DURATION = 0.030
CHIRP_SAMPLES = int(round(CHIRP_DURATION * FS))
CHIRP_F0 = 1400.0
CHIRP_F1 = 3600.0
PREAMBLE_REPEAT = 2
PREAMBLE_THRESHOLD = 0.12

QPSK_SYMBOL_RATE = 350.0
QPSK_SPS = int(round(FS / QPSK_SYMBOL_RATE))
QPSK_SYMBOL_RATE = FS / QPSK_SPS
QPSK_ROLLOFF = 0.35
QPSK_SPAN = 8
QPSK_AMPLITUDE = 0.52
QPSK_FC_DEFAULT = 5200.0
QPSK_BANDPASS_MARGIN = 1400.0
QPSK_SYNC_SYMBOLS = 64
QPSK_TX_PAD_SYMBOLS = 8
QPSK_PRE_PAD_SAMPLES = int(round(0.010 * FS))
QPSK_POST_PAD_SAMPLES = int(round(0.020 * FS))
QPSK_LENGTH_BITS = 16
QPSK_LENGTH_REPEAT = 3
QPSK_CRC_BITS = 32
QPSK_MAX_PAYLOAD_BYTES = 256
QPSK_AIR_REPETITIONS = 1
QPSK_AIR_REPEAT_GAP_S = 0.12
QPSK_AIR_REPEAT_GAP_SAMPLES = int(round(QPSK_AIR_REPEAT_GAP_S * FS))
QPSK_SYNC_THRESHOLD = 0.28
QPSK_TIMING_SEARCH_SYMBOLS = 12
QPSK_CARRIER_SEARCH_OFFSETS = [0.0, -200.0, 200.0, -400.0, 400.0, -600.0, 600.0, -800.0, 800.0]
QPSK_CARRIER_MIN = 2800.0
QPSK_CARRIER_MAX = 9000.0


def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def bytes_to_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bit string length must be a multiple of 8")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def int_to_bits(value: int, width: int) -> str:
    return format(value, f"0{width}b")


def bits_to_int(bits: str) -> int:
    return int(bits, 2) if bits else 0


def repeat_bits(bits: str, n_repeat: int) -> str:
    return "".join(bit * n_repeat for bit in bits)


def majority_decode(bits: str, n_repeat: int) -> str:
    out = []
    for i in range(0, len(bits), n_repeat):
        group = bits[i:i + n_repeat]
        if len(group) < n_repeat:
            break
        ones = group.count("1")
        out.append("1" if ones > (len(group) // 2) else "0")
    return "".join(out)


def crc32_bytes(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


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
    if len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    y = np.asarray(x, dtype=np.float32).copy()
    med, sigma = robust_center_scale(y)
    centered = y - med
    big = np.abs(centered) > blank_sigma * sigma
    centered[big] = 0.0
    np.clip(centered, -clip_sigma * sigma, clip_sigma * sigma, out=centered)
    return (centered + med).astype(np.float32)


def _moving_sum(x: np.ndarray, win: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    return (c[win:] - c[:-win]).astype(np.float32)


def _lin_chirp(n_samples: int, f0: float, f1: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / FS
    duration = max(n_samples / FS, EPS)
    k = (f1 - f0) / duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    y = np.sin(phase) * np.hanning(n_samples)
    return (0.95 * y / (np.max(np.abs(y)) + EPS)).astype(np.float32)


UPCHIRP = _lin_chirp(CHIRP_SAMPLES, CHIRP_F0, CHIRP_F1)
DOWNCHIRP = _lin_chirp(CHIRP_SAMPLES, CHIRP_F1, CHIRP_F0)
PREAMBLE_TEMPLATE = np.concatenate([UPCHIRP, DOWNCHIRP] * PREAMBLE_REPEAT).astype(np.float32)
PREAMBLE_LEN = len(PREAMBLE_TEMPLATE)


def preamble_xcorr(signal: np.ndarray) -> Tuple[Optional[int], float]:
    x = np.asarray(signal, dtype=np.float32)
    if len(x) < PREAMBLE_LEN + 8:
        return None, 0.0
    x = clip_impulses(x)
    if HAVE_SCIPY:
        try:
            sos = butter(6, [1000.0, 4200.0], btype="bandpass", fs=FS, output="sos")
            x = sosfiltfilt(sos, x).astype(np.float32)
        except Exception:
            pass
    x = x - np.mean(x)
    tpl = PREAMBLE_TEMPLATE - np.mean(PREAMBLE_TEMPLATE)
    corr = np.correlate(x, tpl, mode="valid")
    tpl_energy = float(np.dot(tpl, tpl)) + EPS
    local_energy = np.maximum(_moving_sum(x * x, PREAMBLE_LEN), 0.0)
    score = np.abs(corr) / (np.sqrt(local_energy * tpl_energy) + EPS)
    idx = int(np.argmax(score))
    return idx, float(score[idx])


def rrc_taps(beta: float, span: int, sps: int) -> np.ndarray:
    n = np.arange(-span * sps, span * sps + 1, dtype=np.float64)
    t = n / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4.0 / math.pi - 1.0)
        elif beta > 0.0 and abs(abs(4.0 * beta * ti) - 1.0) < 1e-9:
            taps[i] = (
                beta / math.sqrt(2.0)
                * (
                    (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
                    + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
                )
            )
        else:
            num = math.sin(math.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * math.cos(math.pi * ti * (1.0 + beta))
            den = math.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / den
    taps /= math.sqrt(np.sum(taps * taps) + EPS)
    return taps.astype(np.float32)


QPSK_RRC = rrc_taps(QPSK_ROLLOFF, QPSK_SPAN, QPSK_SPS)
QPSK_FILTER_DELAY = (len(QPSK_RRC) - 1) // 2

QPSK_GRAY_MAP = {
    "00": (1.0 + 1.0j) / math.sqrt(2.0),
    "01": (-1.0 + 1.0j) / math.sqrt(2.0),
    "11": (-1.0 - 1.0j) / math.sqrt(2.0),
    "10": (1.0 - 1.0j) / math.sqrt(2.0),
}
QPSK_CONST = np.array(list(QPSK_GRAY_MAP.values()), dtype=np.complex64)
QPSK_BITS_BY_INDEX = ["00", "01", "11", "10"]


def bits_to_qpsk_symbols(bits: str) -> np.ndarray:
    if len(bits) % 2 != 0:
        bits += "0"
    return np.asarray([QPSK_GRAY_MAP[bits[i:i + 2]] for i in range(0, len(bits), 2)], dtype=np.complex64)


def qpsk_symbols_to_bits(symbols: np.ndarray) -> str:
    syms = np.asarray(symbols, dtype=np.complex64).reshape(-1)
    if len(syms) == 0:
        return ""
    dist = np.abs(syms[:, None] - QPSK_CONST[None, :]) ** 2
    idx = np.argmin(dist, axis=1)
    return "".join(QPSK_BITS_BY_INDEX[int(i)] for i in idx)


def _qpsk_sync_symbols() -> np.ndarray:
    rng = np.random.default_rng(20260419)
    bits = "".join("1" if b else "0" for b in rng.integers(0, 2, size=QPSK_SYNC_SYMBOLS * 2))
    return bits_to_qpsk_symbols(bits)


QPSK_SYNC = _qpsk_sync_symbols().astype(np.complex64)


def build_qpsk_frame(message: str) -> Dict[str, object]:
    payload = text_to_bytes(message)
    if len(payload) > QPSK_MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload too large: {len(payload)} bytes > {QPSK_MAX_PAYLOAD_BYTES}")
    payload_bits = bytes_to_bits(payload)
    crc_bits = int_to_bits(crc32_bytes(payload), QPSK_CRC_BITS)
    length_bits = int_to_bits(len(payload), QPSK_LENGTH_BITS)
    length_coded = repeat_bits(length_bits, QPSK_LENGTH_REPEAT)
    frame_bits = length_coded + payload_bits + crc_bits
    frame_symbols = bits_to_qpsk_symbols(frame_bits)
    tx_symbols = np.concatenate(
        [
            np.zeros(QPSK_TX_PAD_SYMBOLS, dtype=np.complex64),
            QPSK_SYNC,
            frame_symbols,
            np.zeros(QPSK_TX_PAD_SYMBOLS, dtype=np.complex64),
        ]
    )
    return {
        "payload": payload,
        "payload_bits": payload_bits,
        "crc_bits": crc_bits,
        "length_bits": length_bits,
        "length_coded": length_coded,
        "frame_bits": frame_bits,
        "frame_symbols": frame_symbols,
        "tx_symbols": tx_symbols,
        "payload_len": len(payload),
    }


def upsample_symbols(symbols: np.ndarray, sps: int) -> np.ndarray:
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = np.asarray(symbols, dtype=np.complex64)
    return up


def modulate_qpsk_symbols(symbols: np.ndarray, fc: float) -> np.ndarray:
    up = upsample_symbols(symbols, QPSK_SPS)
    shaped = np.convolve(up, QPSK_RRC, mode="full")
    n = np.arange(len(shaped), dtype=np.float64)
    carrier = np.exp(1j * 2.0 * np.pi * fc * n / FS)
    passband = QPSK_AMPLITUDE * np.real(shaped * carrier)
    return np.asarray(passband, dtype=np.float32)


def build_qpsk_frame_waveform(message: str, fc: float = QPSK_FC_DEFAULT) -> Tuple[np.ndarray, Dict[str, object]]:
    meta = build_qpsk_frame(message)
    payload_wave = modulate_qpsk_symbols(meta["tx_symbols"], fc)
    frame = np.concatenate(
        [
            np.zeros(PRE_GUARD_SAMPLES, dtype=np.float32),
            PREAMBLE_TEMPLATE,
            np.zeros(QPSK_PRE_PAD_SAMPLES, dtype=np.float32),
            payload_wave,
            np.zeros(QPSK_POST_PAD_SAMPLES, dtype=np.float32),
        ]
    ).astype(np.float32)
    n_info_bits = len(meta["payload"]) * 8
    meta["duration_s"] = len(frame) / FS
    meta["frame_samples"] = len(frame)
    meta["raw_bps"] = 2.0 * QPSK_SYMBOL_RATE
    meta["effective_bps"] = n_info_bits / max(meta["duration_s"], EPS)
    meta["scheme"] = "qpsk"
    meta["fc"] = fc
    return frame, meta


def build_qpsk_air_waveform(message: str, fc: float = QPSK_FC_DEFAULT, air_repetitions: int = QPSK_AIR_REPETITIONS, gap_samples: int = QPSK_AIR_REPEAT_GAP_SAMPLES) -> Tuple[np.ndarray, Dict[str, object]]:
    frame, meta = build_qpsk_frame_waveform(message, fc=fc)
    if air_repetitions <= 1:
        return frame, meta
    pieces = []
    for i in range(air_repetitions):
        pieces.append(frame)
        if i != air_repetitions - 1:
            pieces.append(np.zeros(gap_samples, dtype=np.float32))
    air = np.concatenate(pieces).astype(np.float32)
    meta = dict(meta)
    meta["air_repetitions"] = air_repetitions
    meta["air_duration_s"] = len(air) / FS
    meta["effective_air_bps"] = (len(meta["payload"]) * 8) / max(meta["air_duration_s"], EPS)
    return air, meta


def maybe_bandpass_qpsk(x: np.ndarray, fc: float) -> np.ndarray:
    if not HAVE_SCIPY or len(x) < 64:
        return np.asarray(x, dtype=np.float32)
    low = max(200.0, fc - QPSK_BANDPASS_MARGIN)
    high = min(0.49 * FS, fc + QPSK_BANDPASS_MARGIN)
    try:
        sos = butter(6, [low, high], btype="bandpass", fs=FS, output="sos")
        return sosfiltfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def qpsk_downconvert_real(x: np.ndarray, fc: float) -> np.ndarray:
    n = np.arange(len(x), dtype=np.float64)
    mix = np.exp(-1j * 2.0 * np.pi * fc * n / FS)
    return np.asarray(x, dtype=np.float32) * mix.astype(np.complex64)


def qpsk_matched_filter(bb: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(bb, dtype=np.complex64), QPSK_RRC.astype(np.complex64), mode="full")


def qpsk_sync_metric(rx_sync: np.ndarray) -> float:
    denom = float(np.sum(np.abs(rx_sync)) + EPS)
    return float(np.abs(np.vdot(QPSK_SYNC, rx_sync)) / denom)


def refine_qpsk_timing(mf: np.ndarray, nominal: int) -> Tuple[Optional[int], float, Optional[np.ndarray]]:
    best_offset = None
    best_score = -1.0
    best_sync = None
    span = QPSK_TIMING_SEARCH_SYMBOLS * QPSK_SPS
    lo = max(0, nominal - span)
    hi = min(len(mf) - (QPSK_SYNC_SYMBOLS - 1) * QPSK_SPS - 1, nominal + span)
    for offset in range(lo, hi + 1):
        idx = offset + np.arange(QPSK_SYNC_SYMBOLS) * QPSK_SPS
        if idx[-1] >= len(mf):
            break
        rx_sync = mf[idx]
        score = qpsk_sync_metric(rx_sync)
        if score > best_score:
            best_score = score
            best_offset = offset
            best_sync = rx_sync
    return best_offset, float(best_score), best_sync


def fit_qpsk_phase_ramp(rx_sync: np.ndarray, known_sync: np.ndarray) -> Tuple[float, float]:
    err = np.unwrap(np.angle(rx_sync * np.conj(known_sync)))
    n = np.arange(len(err), dtype=np.float64)
    if len(n) < 2:
        return 0.0, float(np.mean(err))
    slope, intercept = np.polyfit(n, err, 1)
    return float(slope), float(intercept)


def qpsk_apply_phase_correction(symbols: np.ndarray, slope: float, intercept: float, start_index: int = 0) -> np.ndarray:
    n = np.arange(len(symbols), dtype=np.float64) + start_index
    rot = np.exp(-1j * (slope * n + intercept))
    return np.asarray(symbols, dtype=np.complex64) * rot.astype(np.complex64)


def qpsk_decode_length(bits: str) -> Optional[int]:
    needed = QPSK_LENGTH_BITS * QPSK_LENGTH_REPEAT
    if len(bits) < needed:
        return None
    plain = majority_decode(bits[:needed], QPSK_LENGTH_REPEAT)
    if len(plain) != QPSK_LENGTH_BITS:
        return None
    value = bits_to_int(plain)
    if value < 0 or value > QPSK_MAX_PAYLOAD_BYTES:
        return None
    return value




def normalize_carrier_candidates(carriers: list[float] | tuple[float, ...] | np.ndarray) -> list[float]:
    out: list[float] = []
    for fc in carriers:
        f = float(fc)
        if f < QPSK_CARRIER_MIN or f > QPSK_CARRIER_MAX:
            continue
        if any(abs(f - prev) < 1.0 for prev in out):
            continue
        out.append(f)
    return out


def carrier_candidates(center: float) -> list[float]:
    return normalize_carrier_candidates([center + d for d in QPSK_CARRIER_SEARCH_OFFSETS])


def load_carrier_candidates(value: float = QPSK_FC_DEFAULT, carrier_file: str | None = None) -> list[float]:
    center = float(value)
    if carrier_file is not None:
        try:
            with Path(carrier_file).expanduser().resolve().open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if "recommendation" in obj and "qpsk_fc" in obj["recommendation"]:
                center = float(obj["recommendation"]["qpsk_fc"])
            elif "qpsk_fc" in obj:
                center = float(obj["qpsk_fc"])
        except Exception:
            pass
    candidates = carrier_candidates(center)
    if center not in candidates:
        candidates = normalize_carrier_candidates([center] + candidates)
    return candidates


def decode_qpsk_single(signal: np.ndarray, fc: float = QPSK_FC_DEFAULT) -> Dict[str, object]:
    x = np.asarray(signal, dtype=np.float32)
    preamble_start, preamble_score = preamble_xcorr(x)
    if preamble_start is None or preamble_score < PREAMBLE_THRESHOLD:
        return {"ok": False, "reason": "no_preamble", "preamble_score": preamble_score, "scheme": "qpsk"}

    start = preamble_start + PREAMBLE_LEN
    if start >= len(x):
        return {"ok": False, "reason": "need_more", "preamble_score": preamble_score, "scheme": "qpsk"}

    x2 = maybe_bandpass_qpsk(clip_impulses(x[start:]), fc)
    bb = qpsk_downconvert_real(x2, fc)
    mf = qpsk_matched_filter(bb)

    nominal = QPSK_PRE_PAD_SAMPLES + 2 * QPSK_FILTER_DELAY + QPSK_TX_PAD_SYMBOLS * QPSK_SPS
    best_offset, score, rx_sync = refine_qpsk_timing(mf, nominal)
    if best_offset is None or score < QPSK_SYNC_THRESHOLD or rx_sync is None:
        return {"ok": False, "reason": "bad_sync", "preamble_score": preamble_score, "sync_score": score, "scheme": "qpsk"}

    slope, intercept = fit_qpsk_phase_ramp(rx_sync, QPSK_SYNC)
    rx_sync_corr = qpsk_apply_phase_correction(rx_sync, slope, intercept, start_index=0)
    gain = np.vdot(QPSK_SYNC, rx_sync_corr) / (np.vdot(QPSK_SYNC, QPSK_SYNC) + EPS)
    if abs(gain) < 1e-6:
        gain = 1.0 + 0.0j

    total_available = max(0, (len(mf) - best_offset + QPSK_SPS - 1) // QPSK_SPS)
    sample_idx = best_offset + np.arange(total_available) * QPSK_SPS
    sample_idx = sample_idx[sample_idx < len(mf)]
    rx_syms = mf[sample_idx]
    rx_syms = qpsk_apply_phase_correction(rx_syms, slope, intercept, start_index=0) / gain

    data_syms = rx_syms[QPSK_SYNC_SYMBOLS:]
    length_sym_count = int(math.ceil((QPSK_LENGTH_BITS * QPSK_LENGTH_REPEAT) / 2))
    if len(data_syms) < length_sym_count:
        return {"ok": False, "reason": "need_more", "preamble_score": preamble_score, "sync_score": score, "scheme": "qpsk"}

    length_bits = qpsk_symbols_to_bits(data_syms[:length_sym_count])
    payload_len = qpsk_decode_length(length_bits)
    if payload_len is None:
        return {"ok": False, "reason": "bad_length", "preamble_score": preamble_score, "sync_score": score, "scheme": "qpsk"}

    frame_bit_count = QPSK_LENGTH_BITS * QPSK_LENGTH_REPEAT + payload_len * 8 + QPSK_CRC_BITS
    total_data_syms_needed = int(math.ceil(frame_bit_count / 2))
    if len(data_syms) < total_data_syms_needed:
        return {"ok": False, "reason": "need_more", "preamble_score": preamble_score, "sync_score": score, "payload_len": payload_len, "scheme": "qpsk"}

    frame_bits = qpsk_symbols_to_bits(data_syms[:total_data_syms_needed])[:frame_bit_count]
    payload_bits = frame_bits[QPSK_LENGTH_BITS * QPSK_LENGTH_REPEAT : QPSK_LENGTH_BITS * QPSK_LENGTH_REPEAT + payload_len * 8]
    crc_bits = frame_bits[QPSK_LENGTH_BITS * QPSK_LENGTH_REPEAT + payload_len * 8 : QPSK_LENGTH_BITS * QPSK_LENGTH_REPEAT + payload_len * 8 + QPSK_CRC_BITS]

    try:
        payload = bits_to_bytes(payload_bits)
    except Exception:
        return {"ok": False, "reason": "bad_bytes", "preamble_score": preamble_score, "sync_score": score, "payload_len": payload_len, "scheme": "qpsk"}

    rx_crc = bits_to_int(crc_bits)
    calc_crc = crc32_bytes(payload)
    if rx_crc != calc_crc:
        return {"ok": False, "reason": "crc_fail", "preamble_score": preamble_score, "sync_score": score, "payload_len": payload_len, "scheme": "qpsk"}

    return {
        "ok": True,
        "reason": "ok",
        "message": bytes_to_text(payload),
        "payload_len": payload_len,
        "preamble_score": preamble_score,
        "sync_score": score,
        "scheme": "qpsk",
        "fc": fc,
        "timing_method": "matched filter + integer symbol-phase search on known sync",
        "carrier_method": "pilot-aided phase-ramp fit on known sync",
        "constellation": data_syms[:total_data_syms_needed].astype(np.complex64),
        "constellation_corrected": rx_syms[QPSK_SYNC_SYMBOLS:QPSK_SYNC_SYMBOLS + total_data_syms_needed].astype(np.complex64),
        "raw_bps": 2.0 * QPSK_SYMBOL_RATE,
        "effective_bps": (payload_len * 8) / max(len(x) / FS, EPS),
    }




def decode_qpsk_from_signal(signal: np.ndarray, fc: float = QPSK_FC_DEFAULT, carriers: list[float] | None = None) -> Dict[str, object]:
    candidates = carriers if carriers is not None else carrier_candidates(fc)
    best: Dict[str, object] = {"ok": False, "reason": "bad_sync", "preamble_score": 0.0, "sync_score": -1.0, "scheme": "qpsk"}
    for cand in candidates:
        result = decode_qpsk_single(signal, fc=float(cand))
        if result.get("ok"):
            return result
        cand_score = float(result.get("sync_score", -1.0)) + 0.1 * float(result.get("preamble_score", 0.0))
        best_score = float(best.get("sync_score", -1.0)) + 0.1 * float(best.get("preamble_score", 0.0))
        if cand_score > best_score:
            best = result
    return best


def pretty_stats(message: str, meta: Dict[str, object]) -> str:
    payload_len = len(text_to_bytes(message))
    duration = float(meta.get("air_duration_s", meta.get("duration_s", 0.0)))
    raw_bps = float(meta.get("raw_bps", 0.0))
    eff_bps = float(meta.get("effective_air_bps", meta.get("effective_bps", 0.0)))
    return (
        f"scheme=qpsk msg_bytes={payload_len} fc={float(meta.get('fc', QPSK_FC_DEFAULT)):.0f}Hz "
        f"raw_bps={raw_bps:.1f} eff_bps={eff_bps:.1f} air_dur={duration:.3f}s"
    )


def plot_constellation(points: np.ndarray, path: str, title: str = "QPSK Constellation") -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pts = np.asarray(points, dtype=np.complex64).reshape(-1)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(np.real(pts), np.imag(pts), s=14, alpha=0.65)
    ref = np.array(list(QPSK_GRAY_MAP.values()))
    ax.scatter(np.real(ref), np.imag(ref), marker="x", s=80)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    lim = max(1.8, float(np.max(np.abs(np.concatenate([np.real(pts), np.imag(pts), np.real(ref), np.imag(ref)])))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_json(obj: Dict[str, object], path: str) -> str:
    clean = {}
    for k, v in obj.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, sort_keys=True)
    return path
