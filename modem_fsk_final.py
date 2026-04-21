import math
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Robust FSK modem. This branch is optimized for ugly, changing rooms.
# Main idea in v6:
# - keep the room-robust frame structure from v5
# - spend more compute offline after capture instead of trying to decode live
# - make blind sync search stronger because we no longer have real-time pressure
# ---------------------------------------------------------------------------
SAMPLES_PER_SYMBOL = 441
SYMBOL_DURATION = SAMPLES_PER_SYMBOL / FS
AMPLITUDE = 0.55
DEFAULT_TX_GAIN = 0.85

PRE_GUARD_S = 0.050
POST_GUARD_S = 0.030
PRE_GUARD_SAMPLES = int(round(PRE_GUARD_S * FS))
POST_GUARD_SAMPLES = int(round(POST_GUARD_S * FS))

CHIRP_DURATION = 0.030
CHIRP_SAMPLES = int(round(CHIRP_DURATION * FS))
CHIRP_F0 = 1400.0
CHIRP_F1 = 3600.0
PREAMBLE_REPEAT = 2

SYNC_REPEAT = 2
SYNC_ORDER = 6
SYNC_BITS = None
SYNC_LEN = None

LENGTH_BITS = 16
LENGTH_REPEAT = 5
CRC_BITS = 32
PAYLOAD_REPEAT = 3
INTERLEAVER_DEPTH = 8
MAX_PAYLOAD_BYTES = 512

# Frequency diversity. Each full frame is sent in multiple bands.
AIR_REPETITIONS = 1
AIR_REPEAT_GAP_S = 0.180
AIR_REPEAT_GAP_SAMPLES = int(round(AIR_REPEAT_GAP_S * FS))
FRAME_GAP_S = 0.060
FRAME_GAP_SAMPLES = int(round(FRAME_GAP_S * FS))

CLIP_SIGMA = 5.0
BLANK_SIGMA = 9.0
BANDPASS_ORDER = 6

# Low threshold on purpose. False alarms are cheap because sync + CRC reject junk.
PREAMBLE_SCORE_THRESHOLD = 0.10
SYNC_SCORE_THRESHOLD = 0.28
OFFSET_SEARCH = 220

DEFAULT_BLOCKSIZE = 4096
MAX_BUFFER_SECONDS = 45.0
MAX_BUFFER_SAMPLES = int(MAX_BUFFER_SECONDS * FS)
SEARCH_WINDOW_SECONDS = 12.0
SEARCH_WINDOW_SAMPLES = int(SEARCH_WINDOW_SECONDS * FS)

SYNC_PHASE_STEP = 9
BLIND_SYNC_TOPK = 24
BLIND_SYNC_MIN_SCORE = 0.14

WINDOW = np.hanning(SAMPLES_PER_SYMBOL).astype(np.float32)
if not np.any(WINDOW):
    WINDOW = np.ones(SAMPLES_PER_SYMBOL, dtype=np.float32)


@dataclass(frozen=True)
class FSKProfile:
    name: str
    tone0: float
    tone1: float
    band_low: float
    band_high: float


FSK_PROFILES: List[FSKProfile] = [
    FSKProfile("low", 1900.0, 3100.0, 1200.0, 3800.0),
    FSKProfile("high", 6600.0, 7800.0, 5600.0, 8800.0),
]


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


def block_interleave(bits: str, depth: int) -> str:
    if depth <= 1 or not bits:
        return bits
    width = int(math.ceil(len(bits) / depth))
    padded = bits.ljust(depth * width, "0")
    mat = np.frombuffer(padded.encode("ascii"), dtype=np.uint8).reshape(depth, width)
    return mat.T.reshape(-1).tobytes().decode("ascii")


def block_deinterleave(bits: str, depth: int, original_len: int) -> str:
    if depth <= 1 or not bits:
        return bits[:original_len]
    width = int(math.ceil(original_len / depth))
    total = depth * width
    padded = bits.ljust(total, "0")
    mat = np.frombuffer(padded.encode("ascii"), dtype=np.uint8).reshape(width, depth).T
    out = mat.reshape(-1).tobytes().decode("ascii")
    return out[:original_len]


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


def clip_impulses(x: np.ndarray, clip_sigma: float = CLIP_SIGMA, blank_sigma: float = BLANK_SIGMA) -> np.ndarray:
    if len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    y = np.asarray(x, dtype=np.float32).copy()
    med, sigma = robust_center_scale(y)
    centered = y - med
    blank_thr = blank_sigma * sigma
    clip_thr = clip_sigma * sigma
    big = np.abs(centered) > blank_thr
    centered[big] = 0.0
    np.clip(centered, -clip_thr, clip_thr, out=centered)
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


def _mseq(order: int = 6) -> str:
    if order == 6:
        taps = (5, 0)  # x^6 + x + 1
    else:
        raise ValueError("unsupported m-sequence order")
    state = [1] * order
    out = []
    for _ in range(2 ** order - 1):
        out.append(state[-1])
        new_bit = 0
        for idx in taps:
            new_bit ^= state[idx]
        state = [new_bit] + state[:-1]
    return "".join("1" if b else "0" for b in out)


def _init_sync_bits() -> None:
    global SYNC_BITS, SYNC_LEN
    if SYNC_BITS is None:
        period = _mseq(SYNC_ORDER)
        SYNC_BITS = period * SYNC_REPEAT
        SYNC_LEN = len(SYNC_BITS)


_init_sync_bits()

UPCHIRP = _lin_chirp(CHIRP_SAMPLES, CHIRP_F0, CHIRP_F1)
DOWNCHIRP = _lin_chirp(CHIRP_SAMPLES, CHIRP_F1, CHIRP_F0)
PREAMBLE_TEMPLATE = np.concatenate([UPCHIRP, DOWNCHIRP] * PREAMBLE_REPEAT).astype(np.float32)
PREAMBLE_LEN = len(PREAMBLE_TEMPLATE)

_TONE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
_FILTER_CACHE: Dict[str, Optional[np.ndarray]] = {}


def _get_tone_refs(profile: FSKProfile) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = _TONE_CACHE.get(profile.name)
    if cached is not None:
        return cached
    t = np.arange(SAMPLES_PER_SYMBOL, dtype=np.float64) / FS
    sin0 = (np.sin(2.0 * np.pi * profile.tone0 * t) * WINDOW).astype(np.float32)
    cos0 = (np.cos(2.0 * np.pi * profile.tone0 * t) * WINDOW).astype(np.float32)
    sin1 = (np.sin(2.0 * np.pi * profile.tone1 * t) * WINDOW).astype(np.float32)
    cos1 = (np.cos(2.0 * np.pi * profile.tone1 * t) * WINDOW).astype(np.float32)
    _TONE_CACHE[profile.name] = (sin0, cos0, sin1, cos1)
    return _TONE_CACHE[profile.name]


def _get_bandpass(profile: FSKProfile) -> Optional[np.ndarray]:
    cached = _FILTER_CACHE.get(profile.name)
    if cached is not None or profile.name in _FILTER_CACHE:
        return cached
    if not HAVE_SCIPY:
        _FILTER_CACHE[profile.name] = None
        return None
    sos = butter(
        BANDPASS_ORDER,
        [profile.band_low, profile.band_high],
        btype="bandpass",
        fs=FS,
        output="sos",
    )
    _FILTER_CACHE[profile.name] = sos
    return sos


def maybe_bandpass_profile(x: np.ndarray, profile: FSKProfile) -> np.ndarray:
    sos = _get_bandpass(profile)
    if sos is None or len(x) < 64:
        return np.asarray(x, dtype=np.float32)
    try:
        return sosfiltfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def maybe_bandpass_preamble(x: np.ndarray) -> np.ndarray:
    if not HAVE_SCIPY or len(x) < 64:
        return np.asarray(x, dtype=np.float32)
    try:
        sos = butter(6, [1000.0, 4200.0], btype="bandpass", fs=FS, output="sos")
        return sosfiltfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def modulate_fsk(bits: str, profile: FSKProfile) -> np.ndarray:
    if not bits:
        return np.zeros(0, dtype=np.float32)
    sin0, _cos0, sin1, _cos1 = _get_tone_refs(profile)
    sym0 = (AMPLITUDE * sin0).astype(np.float32)
    sym1 = (AMPLITUDE * sin1).astype(np.float32)
    out = np.empty(len(bits) * SAMPLES_PER_SYMBOL, dtype=np.float32)
    for i, bit in enumerate(bits):
        start = i * SAMPLES_PER_SYMBOL
        out[start:start + SAMPLES_PER_SYMBOL] = sym1 if bit == "1" else sym0
    return out


def build_frame_bits(message: str) -> Dict[str, object]:
    payload = text_to_bytes(message)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload too large: {len(payload)} bytes > {MAX_PAYLOAD_BYTES}")
    payload_bits = bytes_to_bits(payload)
    crc_bits = int_to_bits(crc32_bytes(payload), CRC_BITS)
    base_bits = payload_bits + crc_bits
    payload_repeated = repeat_bits(base_bits, PAYLOAD_REPEAT)
    payload_interleaved = block_interleave(payload_repeated, INTERLEAVER_DEPTH)
    length_bits = int_to_bits(len(payload), LENGTH_BITS)
    length_coded = repeat_bits(length_bits, LENGTH_REPEAT)
    frame_bits = SYNC_BITS + length_coded + payload_interleaved
    return {
        "payload": payload,
        "payload_bits": payload_bits,
        "crc_bits": crc_bits,
        "length_bits": length_bits,
        "length_coded": length_coded,
        "payload_coded": payload_interleaved,
        "frame_bits": frame_bits,
        "base_bits": base_bits,
    }


def build_frame_waveform(message: str, profile: FSKProfile) -> Tuple[np.ndarray, Dict[str, object]]:
    meta = build_frame_bits(message)
    fsk = modulate_fsk(meta["frame_bits"], profile)
    frame = np.concatenate(
        [
            np.zeros(PRE_GUARD_SAMPLES, dtype=np.float32),
            PREAMBLE_TEMPLATE,
            fsk,
            np.zeros(POST_GUARD_SAMPLES, dtype=np.float32),
        ]
    ).astype(np.float32)
    meta = dict(meta)
    meta["frame_samples"] = len(frame)
    meta["duration_s"] = len(frame) / FS
    meta["profile"] = profile.name
    meta["scheme"] = "fsk"
    meta["raw_bps"] = 1.0 / SYMBOL_DURATION
    meta["effective_bps"] = (len(meta["payload"]) * 8) / max(meta["duration_s"], EPS)
    return frame, meta


def build_air_waveform(message: str, air_repetitions: int = AIR_REPETITIONS, gap_samples: int = AIR_REPEAT_GAP_SAMPLES) -> Tuple[np.ndarray, Dict[str, object]]:
    frames = []
    last_meta = None
    profile_names: List[str] = []
    for rep in range(air_repetitions):
        for p_idx, profile in enumerate(FSK_PROFILES):
            frame, meta = build_frame_waveform(message, profile)
            frames.append(frame)
            profile_names.append(profile.name)
            last_meta = meta
            if not (rep == air_repetitions - 1 and p_idx == len(FSK_PROFILES) - 1):
                frames.append(np.zeros(FRAME_GAP_SAMPLES, dtype=np.float32))
        if rep != air_repetitions - 1:
            frames.append(np.zeros(gap_samples, dtype=np.float32))
    air = np.concatenate(frames).astype(np.float32)
    meta = dict(last_meta or {})
    meta["air_repetitions"] = air_repetitions
    meta["profiles"] = profile_names
    meta["air_duration_s"] = len(air) / FS
    meta["effective_air_bps"] = (len(text_to_bytes(message)) * 8) / max(meta["air_duration_s"], EPS)
    return air, meta


def preamble_xcorr(signal: np.ndarray, search_only_tail: bool = False) -> Tuple[Optional[int], float]:
    x = np.asarray(signal, dtype=np.float32)
    if len(x) < PREAMBLE_LEN + 8:
        return None, 0.0
    if search_only_tail and len(x) > SEARCH_WINDOW_SAMPLES:
        base = len(x) - SEARCH_WINDOW_SAMPLES
        x = x[base:]
    else:
        base = 0
    x = clip_impulses(x)
    x = maybe_bandpass_preamble(x)
    x = x - np.mean(x)
    tpl = PREAMBLE_TEMPLATE.astype(np.float32)
    tpl = tpl - np.mean(tpl)
    corr = np.correlate(x, tpl, mode="valid")
    tpl_energy = float(np.dot(tpl, tpl)) + EPS
    sq = x * x
    local_energy = np.maximum(_moving_sum(sq, PREAMBLE_LEN), 0.0)
    energy = np.sqrt(local_energy * tpl_energy) + EPS
    score = np.abs(corr) / energy
    local_rms = np.sqrt(local_energy / PREAMBLE_LEN + EPS)
    score[local_rms < 0.003] = 0.0
    idx = int(np.argmax(score))
    best = float(score[idx])
    return base + idx, best


def demod_soft_metrics(x: np.ndarray, start: int, n_bits: int, profile: FSKProfile) -> Optional[np.ndarray]:
    needed = start + n_bits * SAMPLES_PER_SYMBOL
    if start < 0 or needed > len(x):
        return None
    segs = np.asarray(x[start:needed], dtype=np.float32).reshape(n_bits, SAMPLES_PER_SYMBOL)
    segs = segs - np.mean(segs, axis=1, keepdims=True)
    sin0, cos0, sin1, cos1 = _get_tone_refs(profile)
    i0 = segs @ cos0
    q0 = segs @ sin0
    i1 = segs @ cos1
    q1 = segs @ sin1
    e0 = i0 * i0 + q0 * q0
    e1 = i1 * i1 + q1 * q1
    return (e1 - e0).astype(np.float32)


def soft_to_hard(soft: np.ndarray) -> str:
    return "".join("1" if v > 0.0 else "0" for v in soft)


def expected_sync_signs() -> np.ndarray:
    return np.array([1.0 if b == "1" else -1.0 for b in SYNC_BITS], dtype=np.float32)


SYNC_SIGNS = expected_sync_signs()


def sync_score(soft_sync: np.ndarray) -> float:
    denom = float(np.sum(np.abs(soft_sync)) + EPS)
    return float(np.dot(soft_sync, SYNC_SIGNS) / denom)


def decode_length_bits(length_hard: str) -> Optional[int]:
    if len(length_hard) < LENGTH_BITS * LENGTH_REPEAT:
        return None
    plain = majority_decode(length_hard[: LENGTH_BITS * LENGTH_REPEAT], LENGTH_REPEAT)
    if len(plain) != LENGTH_BITS:
        return None
    value = bits_to_int(plain)
    if value < 0 or value > MAX_PAYLOAD_BYTES:
        return None
    return value


def needed_payload_coded_bits(payload_len_bytes: int) -> int:
    return (payload_len_bytes * 8 + CRC_BITS) * PAYLOAD_REPEAT


def total_frame_bits_for_payload(payload_len_bytes: int) -> int:
    return SYNC_LEN + LENGTH_BITS * LENGTH_REPEAT + needed_payload_coded_bits(payload_len_bytes)


def total_samples_for_payload(payload_len_bytes: int) -> int:
    return PRE_GUARD_SAMPLES + PREAMBLE_LEN + total_frame_bits_for_payload(payload_len_bytes) * SAMPLES_PER_SYMBOL + POST_GUARD_SAMPLES


def refine_fsk_start(x: np.ndarray, nominal_fsk_start: int, profile: FSKProfile) -> Tuple[int, float]:
    best_start = nominal_fsk_start
    best_score = -1.0
    for delta in range(-OFFSET_SEARCH, OFFSET_SEARCH + 1, 4):
        start = nominal_fsk_start + delta
        soft = demod_soft_metrics(x, start, SYNC_LEN, profile)
        if soft is None:
            continue
        score = sync_score(soft)
        if score > best_score:
            best_score = score
            best_start = start
    for delta in range(-3, 4):
        start = best_start + delta
        soft = demod_soft_metrics(x, start, SYNC_LEN, profile)
        if soft is None:
            continue
        score = sync_score(soft)
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_score


def decode_frame_from_preamble_start(signal: np.ndarray, preamble_start: int, allowed_profiles: list[FSKProfile] | None = None) -> Dict[str, object]:
    x = np.asarray(signal, dtype=np.float32)
    nominal_fsk_start = preamble_start + PREAMBLE_LEN
    search_left = max(0, nominal_fsk_start - OFFSET_SEARCH)
    best_fail: Dict[str, object] = {"ok": False, "reason": "bad_sync", "sync_score": -1.0, "scheme": "fsk"}

    profiles = allowed_profiles if allowed_profiles else FSK_PROFILES
    for profile in profiles:
        x2 = maybe_bandpass_profile(clip_impulses(x[search_left:]), profile)
        local_nominal = nominal_fsk_start - search_left
        fsk_start_local, score = refine_fsk_start(x2, local_nominal, profile)
        if score > float(best_fail.get("sync_score", -1.0)):
            best_fail = {
                "ok": False,
                "reason": "bad_sync",
                "sync_score": score,
                "scheme": "fsk",
                "profile": profile.name,
            }
        if score < SYNC_SCORE_THRESHOLD:
            continue

        soft_hdr = demod_soft_metrics(
            x2,
            fsk_start_local + SYNC_LEN * SAMPLES_PER_SYMBOL,
            LENGTH_BITS * LENGTH_REPEAT,
            profile,
        )
        if soft_hdr is None:
            continue
        hdr_bits = soft_to_hard(soft_hdr)
        payload_len = decode_length_bits(hdr_bits)
        if payload_len is None:
            continue

        n_payload_coded = needed_payload_coded_bits(payload_len)
        soft_payload = demod_soft_metrics(
            x2,
            fsk_start_local + (SYNC_LEN + LENGTH_BITS * LENGTH_REPEAT) * SAMPLES_PER_SYMBOL,
            n_payload_coded,
            profile,
        )
        if soft_payload is None:
            continue

        hard_payload = soft_to_hard(soft_payload)
        deint = block_deinterleave(hard_payload, INTERLEAVER_DEPTH, n_payload_coded)
        base_bits = majority_decode(deint, PAYLOAD_REPEAT)
        expected_base_len = payload_len * 8 + CRC_BITS
        if len(base_bits) < expected_base_len:
            continue
        base_bits = base_bits[:expected_base_len]
        payload_bits = base_bits[: payload_len * 8]
        crc_bits = base_bits[payload_len * 8 : payload_len * 8 + CRC_BITS]

        try:
            payload = bits_to_bytes(payload_bits)
        except Exception:
            continue

        rx_crc = bits_to_int(crc_bits)
        calc_crc = crc32_bytes(payload)
        if rx_crc != calc_crc:
            best_fail = {
                "ok": False,
                "reason": "crc_fail",
                "sync_score": score,
                "scheme": "fsk",
                "profile": profile.name,
                "payload_len": payload_len,
            }
            continue

        message = bytes_to_text(payload)
        total_samples = total_samples_for_payload(payload_len)
        duration_s = total_samples / FS
        return {
            "ok": True,
            "reason": "ok",
            "message": message,
            "payload_len": payload_len,
            "sync_score": score,
            "preamble_start": preamble_start,
            "fsk_start": search_left + fsk_start_local,
            "frame_samples": total_samples,
            "payload": payload,
            "duration_s": duration_s,
            "raw_bps": 1.0 / SYMBOL_DURATION,
            "effective_bps": (payload_len * 8) / max(duration_s, EPS),
            "scheme": "fsk",
            "profile": profile.name,
            "timing_method": "preamble xcorr + sync-sequence offset search",
        }

    return best_fail


def _soft_metrics_all_phases_preprocessed(x2: np.ndarray, profile: FSKProfile, phase: int) -> Optional[np.ndarray]:
    n_syms = (len(x2) - phase) // SAMPLES_PER_SYMBOL
    if n_syms < SYNC_LEN + LENGTH_BITS * LENGTH_REPEAT:
        return None
    segs = x2[phase: phase + n_syms * SAMPLES_PER_SYMBOL].reshape(n_syms, SAMPLES_PER_SYMBOL)
    segs = segs - np.mean(segs, axis=1, keepdims=True)
    sin0, cos0, sin1, cos1 = _get_tone_refs(profile)
    i0 = segs @ cos0
    q0 = segs @ sin0
    i1 = segs @ cos1
    q1 = segs @ sin1
    e0 = i0 * i0 + q0 * q0
    e1 = i1 * i1 + q1 * q1
    return (e1 - e0).astype(np.float32)


def blind_sync_candidates(signal: np.ndarray, search_only_tail: bool = False, top_k: int = BLIND_SYNC_TOPK, allowed_profiles: list[FSKProfile] | None = None) -> List[Dict[str, object]]:
    x = np.asarray(signal, dtype=np.float32)
    if search_only_tail and len(x) > SEARCH_WINDOW_SAMPLES:
        base = len(x) - SEARCH_WINDOW_SAMPLES
        x = x[base:]
    else:
        base = 0

    candidates: List[Dict[str, object]] = []
    clipped = clip_impulses(x)
    profiles = allowed_profiles if allowed_profiles else FSK_PROFILES
    for profile in profiles:
        x2 = maybe_bandpass_profile(clipped, profile)
        local: List[Tuple[float, int]] = []
        for phase in range(0, SAMPLES_PER_SYMBOL, SYNC_PHASE_STEP):
            soft = _soft_metrics_all_phases_preprocessed(x2, profile, phase)
            if soft is None or len(soft) < SYNC_LEN:
                continue

            num = np.correlate(soft, SYNC_SIGNS, mode="valid")
            abs_soft = np.abs(soft).astype(np.float64)
            csum = np.cumsum(np.concatenate([[0.0], abs_soft]))
            denom = (csum[SYNC_LEN:] - csum[:-SYNC_LEN]) + EPS
            score = num / denom
            if len(score) == 0:
                continue

            # Greedy local-peak selection.
            order = np.argsort(score)[::-1]
            for idx in order:
                s = float(score[idx])
                if s < BLIND_SYNC_MIN_SCORE:
                    break
                sync_start = base + phase + int(idx) * SAMPLES_PER_SYMBOL
                if any(abs(sync_start - prev_start) < (2 * SAMPLES_PER_SYMBOL) for _prev_s, prev_start in local):
                    continue
                local.append((s, sync_start))
                if len(local) >= max(6, top_k):
                    break

        for s, sync_start in local:
            candidates.append({
                "sync_score": float(s),
                "sync_start": int(sync_start),
                "profile": profile.name,
            })

    candidates.sort(key=lambda d: float(d.get("sync_score", -1.0)), reverse=True)
    deduped: List[Dict[str, object]] = []
    for cand in candidates:
        sync_start = int(cand["sync_start"])
        profile_name = str(cand["profile"])
        if any(profile_name == str(prev["profile"]) and abs(sync_start - int(prev["sync_start"])) < (2 * SAMPLES_PER_SYMBOL) for prev in deduped):
            continue
        deduped.append(cand)
        if len(deduped) >= top_k:
            break
    return deduped


def decode_frame_from_sync_start(signal: np.ndarray, sync_start: int, profile_name: str) -> Dict[str, object]:
    x = np.asarray(signal, dtype=np.float32)
    profile = next((p for p in FSK_PROFILES if p.name == profile_name), None)
    if profile is None:
        return {"ok": False, "reason": "bad_profile", "scheme": "fsk"}

    x2 = maybe_bandpass_profile(clip_impulses(x), profile)

    soft_hdr = demod_soft_metrics(
        x2,
        sync_start + SYNC_LEN * SAMPLES_PER_SYMBOL,
        LENGTH_BITS * LENGTH_REPEAT,
        profile,
    )
    if soft_hdr is None:
        return {"ok": False, "reason": "need_more", "scheme": "fsk", "profile": profile.name}

    hdr_bits = soft_to_hard(soft_hdr)
    payload_len = decode_length_bits(hdr_bits)
    if payload_len is None:
        return {"ok": False, "reason": "bad_length", "scheme": "fsk", "profile": profile.name}

    n_payload_coded = needed_payload_coded_bits(payload_len)
    soft_payload = demod_soft_metrics(
        x2,
        sync_start + (SYNC_LEN + LENGTH_BITS * LENGTH_REPEAT) * SAMPLES_PER_SYMBOL,
        n_payload_coded,
        profile,
    )
    if soft_payload is None:
        return {"ok": False, "reason": "need_more", "scheme": "fsk", "profile": profile.name, "payload_len": payload_len}

    hard_payload = soft_to_hard(soft_payload)
    deint = block_deinterleave(hard_payload, INTERLEAVER_DEPTH, n_payload_coded)
    base_bits = majority_decode(deint, PAYLOAD_REPEAT)
    expected_base_len = payload_len * 8 + CRC_BITS
    if len(base_bits) < expected_base_len:
        return {"ok": False, "reason": "need_more", "scheme": "fsk", "profile": profile.name, "payload_len": payload_len}

    base_bits = base_bits[:expected_base_len]
    payload_bits = base_bits[: payload_len * 8]
    crc_bits = base_bits[payload_len * 8 : payload_len * 8 + CRC_BITS]

    try:
        payload = bits_to_bytes(payload_bits)
    except Exception:
        return {"ok": False, "reason": "bad_bytes", "scheme": "fsk", "profile": profile.name, "payload_len": payload_len}

    rx_crc = bits_to_int(crc_bits)
    calc_crc = crc32_bytes(payload)
    if rx_crc != calc_crc:
        return {"ok": False, "reason": "crc_fail", "scheme": "fsk", "profile": profile.name, "payload_len": payload_len}

    message = bytes_to_text(payload)
    total_samples = total_samples_for_payload(payload_len)
    duration_s = total_samples / FS
    return {
        "ok": True,
        "reason": "ok",
        "message": message,
        "payload_len": payload_len,
        "frame_samples": total_samples,
        "payload": payload,
        "duration_s": duration_s,
        "raw_bps": 1.0 / SYMBOL_DURATION,
        "effective_bps": (payload_len * 8) / max(duration_s, EPS),
        "scheme": "fsk",
        "profile": profile.name,
        "timing_method": "blind sync-sequence search",
    }


def decode_best_from_signal(signal: np.ndarray, search_tail: bool = False, allowed_profiles: list[FSKProfile] | None = None) -> Dict[str, object]:
    preamble_start, preamble_score = preamble_xcorr(signal, search_only_tail=search_tail)
    best_fail: Dict[str, object] = {"ok": False, "reason": "no_preamble", "preamble_score": preamble_score, "scheme": "fsk"}

    if preamble_start is not None and preamble_score >= PREAMBLE_SCORE_THRESHOLD:
        result = decode_frame_from_preamble_start(signal, preamble_start, allowed_profiles=allowed_profiles)
        result["preamble_score"] = preamble_score
        if result.get("ok"):
            return result
        best_fail = result

    blind = blind_sync_candidates(signal, search_only_tail=search_tail, top_k=BLIND_SYNC_TOPK, allowed_profiles=allowed_profiles)
    for cand in blind:
        result = decode_frame_from_sync_start(signal, int(cand["sync_start"]), str(cand["profile"]))
        result["sync_score"] = float(cand["sync_score"])
        result["preamble_score"] = preamble_score
        if result.get("ok"):
            return result
        if float(result.get("sync_score", -1.0)) > float(best_fail.get("sync_score", -1.0)):
            best_fail = result

    if "preamble_score" not in best_fail:
        best_fail["preamble_score"] = preamble_score
    return best_fail


def pretty_stats(message: str, meta: Dict[str, object]) -> str:
    payload_len = len(text_to_bytes(message))
    duration = float(meta.get("air_duration_s", meta.get("duration_s", 0.0)))
    raw_bps = float(meta.get("raw_bps", 0.0))
    eff_bps = float(meta.get("effective_air_bps", meta.get("effective_bps", 0.0)))
    profiles = ",".join(meta.get("profiles", [meta.get("profile", "")]))
    return (
        f"scheme=fsk msg_bytes={payload_len} profiles={profiles} "
        f"raw_bps={raw_bps:.1f} eff_bps={eff_bps:.1f} air_dur={duration:.3f}s"
    )


# ---------------------------------------------------------------------------
# Extra tooling for the professor-facing demo.
# ---------------------------------------------------------------------------
MEASURE_DURATION_S = 2.0
MEASURE_F0 = 500.0
MEASURE_F1 = 10000.0
MEASURE_AMPLITUDE = 0.60
MEASURE_GUARD_S = 0.100
MEASURE_GUARD_SAMPLES = int(round(MEASURE_GUARD_S * FS))
MEASURE_NPERSEG = 4096


def get_profile(name: str) -> FSKProfile:
    for profile in FSK_PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown profile: {name}")


def list_profile_names() -> List[str]:
    return [p.name for p in FSK_PROFILES]


def resolve_profiles(spec: str | None) -> List[FSKProfile]:
    if spec is None or spec.strip() == "" or spec.strip().lower() in {"all", "default", "both"}:
        return list(FSK_PROFILES)
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    profiles: List[FSKProfile] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        profiles.append(get_profile(name))
        seen.add(name)
    if not profiles:
        raise ValueError("no valid profiles selected")
    return profiles




def _load_json_profile_data(path: str | None) -> dict:
    if not path:
        return {}
    import json
    from pathlib import Path
    try:
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def resolve_profile_spec(
    explicit: str | None = None,
    profiles_file: str | None = None,
    channel_file: str | None = None,
    ambient_file: str | None = None,
) -> str:
    exp = (explicit or "").strip()
    if exp and exp.lower() not in {"auto", "default"}:
        return exp

    specs: list[str] = []

    for path in [profiles_file, channel_file, ambient_file]:
        obj = _load_json_profile_data(path)
        if not obj:
            continue
        if isinstance(obj.get("selected_profiles"), list) and obj["selected_profiles"]:
            specs.extend(str(x).lower() for x in obj["selected_profiles"] if str(x).strip())
        rec = obj.get("recommendation", {}) if isinstance(obj.get("recommendation"), dict) else {}
        prof = obj.get("fsk_profile") or rec.get("fsk_profile")
        if prof:
            specs.append(str(prof).lower())

    if not specs:
        return "low,high"

    seen = []
    for name in specs:
        if name not in {"low", "high"}:
            continue
        if name not in seen:
            seen.append(name)

    if not seen:
        return "low,high"
    if len(seen) == 1:
        return seen[0]
    ordered = [name for name in ["low", "high"] if name in seen]
    return ",".join(ordered)
def build_air_waveform_selected(
    message: str,
    selected_profiles: List[FSKProfile],
    air_repetitions: int = AIR_REPETITIONS,
    gap_samples: int = AIR_REPEAT_GAP_SAMPLES,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if not selected_profiles:
        raise ValueError("selected_profiles must not be empty")
    frames = []
    duration_accum = 0
    last_meta = None
    profile_names: List[str] = []
    for rep in range(air_repetitions):
        for p_idx, profile in enumerate(selected_profiles):
            frame, meta = build_frame_waveform(message, profile)
            frames.append(frame)
            duration_accum += len(frame)
            profile_names.append(profile.name)
            last_meta = meta
            if not (rep == air_repetitions - 1 and p_idx == len(selected_profiles) - 1):
                frames.append(np.zeros(FRAME_GAP_SAMPLES, dtype=np.float32))
                duration_accum += FRAME_GAP_SAMPLES
        if rep != air_repetitions - 1:
            frames.append(np.zeros(gap_samples, dtype=np.float32))
            duration_accum += gap_samples
    air = np.concatenate(frames).astype(np.float32)
    meta = dict(last_meta or {})
    meta["air_repetitions"] = air_repetitions
    meta["profiles"] = profile_names
    meta["air_duration_s"] = len(air) / FS
    meta["effective_air_bps"] = (len(text_to_bytes(message)) * 8) / max(meta["air_duration_s"], EPS)
    return air, meta


def _exp_sweep(n_samples: int, f0: float, f1: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / FS
    duration = max(n_samples / FS, EPS)
    ratio = max(f1 / max(f0, EPS), 1.000001)
    k = duration / math.log(ratio)
    phase = 2.0 * np.pi * f0 * k * (np.exp(t / k) - 1.0)
    y = np.sin(phase)
    fade = np.ones_like(y)
    ramp = min(len(y) // 20, int(0.03 * FS))
    if ramp > 1:
        win = np.hanning(2 * ramp)
        fade[:ramp] = win[:ramp]
        fade[-ramp:] = win[ramp:]
    y *= fade
    return (0.95 * y / (np.max(np.abs(y)) + EPS)).astype(np.float32)


def build_measurement_waveform(
    duration_s: float = MEASURE_DURATION_S,
    f0: float = MEASURE_F0,
    f1: float = MEASURE_F1,
    amplitude: float = MEASURE_AMPLITUDE,
) -> Tuple[np.ndarray, Dict[str, object]]:
    n = int(round(duration_s * FS))
    sweep = _exp_sweep(n, f0, f1)
    wave = np.concatenate(
        [
            np.zeros(MEASURE_GUARD_SAMPLES, dtype=np.float32),
            (amplitude * sweep).astype(np.float32),
            np.zeros(MEASURE_GUARD_SAMPLES, dtype=np.float32),
        ]
    ).astype(np.float32)
    meta = {
        "scheme": "sweep",
        "duration_s": len(wave) / FS,
        "sweep_duration_s": duration_s,
        "f0": f0,
        "f1": f1,
        "raw_bps": 0.0,
        "effective_bps": 0.0,
    }
    return wave, meta


def find_measurement_start(recording: np.ndarray, tx_wave: np.ndarray) -> Tuple[Optional[int], float]:
    x = np.asarray(recording, dtype=np.float32)
    tpl = np.asarray(tx_wave, dtype=np.float32)
    if len(x) < len(tpl):
        return None, 0.0
    corr = np.correlate(x, tpl, mode="valid")
    tpl_energy = float(np.dot(tpl, tpl)) + EPS
    local_energy = np.maximum(_moving_sum(x * x, len(tpl)), 0.0)
    score = np.abs(corr) / (np.sqrt(local_energy * tpl_energy) + EPS)
    idx = int(np.argmax(score))
    return idx, float(score[idx])


def _interp_mag(freq: np.ndarray, mag_db: np.ndarray, f_hz: float) -> float:
    if f_hz <= freq[0]:
        return float(mag_db[0])
    if f_hz >= freq[-1]:
        return float(mag_db[-1])
    return float(np.interp(f_hz, freq, mag_db))


def recommend_from_response(freq: np.ndarray, mag_db: np.ndarray) -> Dict[str, object]:
    valid = (freq >= 1200.0) & (freq <= min(9000.0, 0.48 * FS))
    if not np.any(valid):
        return {
            "fsk_profile": "low",
            "fsk_tone0": FSK_PROFILES[0].tone0,
            "fsk_tone1": FSK_PROFILES[0].tone1,
            "qpsk_fc": 5200.0,
            "notes": "response band too small",
        }
    f = freq[valid]
    m = mag_db[valid]
    width = max(5, len(m) // 120)
    kernel = np.ones(width, dtype=np.float64) / width
    sm = np.convolve(m, kernel, mode="same")

    profile_scores = []
    for profile in FSK_PROFILES:
        s0 = _interp_mag(f, sm, profile.tone0)
        s1 = _interp_mag(f, sm, profile.tone1)
        score = min(s0, s1) - 0.15 * abs(s0 - s1)
        profile_scores.append((score, profile, s0, s1))
    profile_scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_profile, s0, s1 = profile_scores[0]

    best_fc = 5200.0
    best_fc_score = -1e9
    beta = 0.35
    symbol_rate = 350.0
    half_bw = 0.5 * symbol_rate * (1.0 + beta) + 250.0
    for fc in np.arange(max(2500.0, f[0] + half_bw), min(8500.0, f[-1] - half_bw), 100.0):
        left = _interp_mag(f, sm, fc - half_bw)
        mid = _interp_mag(f, sm, fc)
        right = _interp_mag(f, sm, fc + half_bw)
        flatness = max(abs(mid - left), abs(mid - right), abs(left - right))
        score = min(left, mid, right) - 0.25 * flatness
        if score > best_fc_score:
            best_fc_score = score
            best_fc = float(fc)

    notes = (
        f"best FSK profile {best_profile.name} tone0={best_profile.tone0:.0f}Hz tone1={best_profile.tone1:.0f}Hz "
        f"tone levels {s0:.1f}/{s1:.1f} dB, recommended QPSK carrier {best_fc:.0f}Hz"
    )
    return {
        "fsk_profile": best_profile.name,
        "fsk_tone0": best_profile.tone0,
        "fsk_tone1": best_profile.tone1,
        "qpsk_fc": best_fc,
        "notes": notes,
    }


def evaluate_scheme_vs_channel(freq: np.ndarray, mag_db: np.ndarray, profile_name: str = "low") -> Dict[str, float | str]:
    profile = get_profile(profile_name)
    m0 = _interp_mag(freq, mag_db, profile.tone0)
    m1 = _interp_mag(freq, mag_db, profile.tone1)
    delta = abs(m0 - m1)
    matters = "yes" if delta > 3.0 else "probably not much"
    return {
        "profile": profile.name,
        "tone0_db": m0,
        "tone1_db": m1,
        "imbalance_db": delta,
        "channel_matters": matters,
    }


def estimate_channel_response(
    recording: np.ndarray,
    duration_s: float = MEASURE_DURATION_S,
    f0: float = MEASURE_F0,
    f1: float = MEASURE_F1,
) -> Dict[str, object]:
    tx_wave, meta = build_measurement_waveform(duration_s=duration_s, f0=f0, f1=f1)
    start, score = find_measurement_start(recording, tx_wave)
    if start is None:
        return {"ok": False, "reason": "too_short", "scheme": "sweep"}
    x = np.asarray(recording, dtype=np.float32)
    if start + len(tx_wave) > len(x):
        return {"ok": False, "reason": "need_more", "scheme": "sweep", "start": start}

    rx_seg = x[start:start + len(tx_wave)]
    tx_seg = tx_wave

    if HAVE_SCIPY and csd is not None and welch is not None:
        nperseg = min(MEASURE_NPERSEG, len(tx_seg))
        freq, pxy = csd(tx_seg, rx_seg, fs=FS, nperseg=nperseg)
        _, pxx = welch(tx_seg, fs=FS, nperseg=nperseg)
        h = pxy / (pxx + EPS)
        if coherence is not None:
            _, coh = coherence(tx_seg, rx_seg, fs=FS, nperseg=nperseg)
        else:
            coh = np.ones_like(freq)
    else:
        nfft = 1
        while nfft < len(tx_seg):
            nfft *= 2
        X = np.fft.rfft(tx_seg, n=nfft)
        Y = np.fft.rfft(rx_seg, n=nfft)
        h = Y / (X + EPS)
        freq = np.fft.rfftfreq(nfft, 1.0 / FS)
        coh = np.ones_like(freq)

    mag = np.abs(h)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-9))
    rec = recommend_from_response(freq, mag_db)
    return {
        "ok": True,
        "scheme": "sweep",
        "reason": "ok",
        "start": start,
        "score": score,
        "tx": tx_seg,
        "rx": rx_seg,
        "freq": freq,
        "mag": mag,
        "mag_db": mag_db,
        "coherence": coh,
        "recommendation": rec,
        "fsk_eval_low": evaluate_scheme_vs_channel(freq, mag_db, "low"),
        "fsk_eval_high": evaluate_scheme_vs_channel(freq, mag_db, "high"),
    }


def ambient_band_powers(recording: np.ndarray) -> Dict[str, float]:
    x = np.asarray(recording, dtype=np.float32)
    if len(x) == 0:
        return {p.name: float("inf") for p in FSK_PROFILES}
    if HAVE_SCIPY and welch is not None:
        nperseg = min(MEASURE_NPERSEG, len(x))
        freq, pxx = welch(x, fs=FS, nperseg=nperseg)
    else:
        nfft = 1
        while nfft < len(x):
            nfft *= 2
        X = np.fft.rfft(x, n=nfft)
        freq = np.fft.rfftfreq(nfft, 1.0 / FS)
        pxx = (np.abs(X) ** 2) / max(len(X), 1)
    out: Dict[str, float] = {}
    for profile in FSK_PROFILES:
        mask = (freq >= profile.band_low) & (freq <= profile.band_high)
        if np.any(mask):
            out[profile.name] = float(10.0 * np.log10(np.mean(pxx[mask]) + EPS))
        else:
            out[profile.name] = float("inf")
    return out


def choose_profiles_from_ambient(recording: np.ndarray, hysteresis_db: float = 4.0) -> Dict[str, object]:
    powers = ambient_band_powers(recording)
    ordered = sorted(powers.items(), key=lambda kv: kv[1])
    best_name, best_db = ordered[0]
    second_name, second_db = ordered[1] if len(ordered) > 1 else (best_name, best_db)
    if second_db - best_db >= hysteresis_db:
        selected = [best_name]
        note = f"ambient noise clearly favors {best_name} by {second_db - best_db:.1f} dB"
    else:
        selected = [name for name, _db in ordered]
        note = f"ambient noise difference only {second_db - best_db:.1f} dB, use diversity"
    return {
        "selected_profiles": selected,
        "band_powers_db": powers,
        "notes": note,
    }


def plot_frequency_response(freq: np.ndarray, mag_db: np.ndarray, path: str, title: str = "Speaker-Mic Magnitude Response") -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(freq, mag_db, linewidth=1.0)
    for profile in FSK_PROFILES:
        ax.axvspan(profile.band_low, profile.band_high, alpha=0.08, label=f"{profile.name} band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(freq[0]), float(freq[-1]))
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_ambient_psd(powers_db: Dict[str, float], path: str, title: str = "Ambient Noise by Band") -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    names = list(powers_db.keys())
    vals = [powers_db[name] for name in names]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(names, vals)
    ax.set_ylabel("Band Power (dB, relative)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_frequency_response_csv(freq: np.ndarray, mag_db: np.ndarray, path: str) -> str:
    data = np.column_stack([freq, mag_db])
    header = "frequency_hz,magnitude_db"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return path


def save_json(obj: Dict[str, object], path: str) -> str:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path
