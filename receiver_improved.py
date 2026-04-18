"""
Improved audio data receiver for laptop-to-laptop communication.

Matches the framing in transmitter_improved.py:

    [ Barker-13 preamble          ]
    [ Length header     16 bits x3 ]
    [ Length header inv 16 bits x3 ]   <-- verified against length
    [ Payload             N bits x3 ]
    [ CRC-16-CCITT       16 bits x3 ]

The inverted-length copy lets us detect a corrupted header reliably,
so we never slice a random number of payload samples and emit garbage.
If the header check fails, the receiver prints a diagnostic and aborts.

Set DEBUG = True to see per-stage dumps (preamble correlation peak,
raw repeated bits before majority vote, decoded header values, etc.).
This is invaluable for figuring out *why* a real-world recording
failed -- header corruption, preamble mis-sync, and clipped audio all
look different in the diagnostic output.
"""

import numpy as np
import sounddevice as sd
from typing import List, Optional, Tuple

from transmitter_improved import (
    FS,
    BIT_DURATION,
    F_MARK,
    F_SPACE,
    AMPLITUDE,
    PREAMBLE_BITS,
    N_REPEAT,
    LENGTH_BITS,
    LENGTH_MASK,
    CRC_BITS,
    crc16_ccitt,
)

RECORD_DURATION = 6.0
DEBUG = True   # flip to False to silence per-stage diagnostics

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
def record_audio(duration: float = RECORD_DURATION) -> np.ndarray:
    print(f"[rx] Recording for {duration} s ...")
    recording = sd.rec(int(duration * FS), samplerate=FS, channels=1, blocking=True)
    return recording.flatten()

# ---------------------------------------------------------------------------
# Waveform templates
# ---------------------------------------------------------------------------
def generate_bit_wave(bit: str) -> np.ndarray:
    t = np.linspace(0, BIT_DURATION, int(FS * BIT_DURATION), endpoint=False)
    freq = F_MARK if bit == '1' else F_SPACE
    tone = np.sin(2 * np.pi * freq * t)
    window = np.hanning(len(t))
    return AMPLITUDE * tone * window

def build_preamble_wave() -> np.ndarray:
    return np.concatenate([generate_bit_wave(b) for b in PREAMBLE_BITS])

# ---------------------------------------------------------------------------
# Synchronisation and demodulation
# ---------------------------------------------------------------------------
def find_preamble_start(recording: np.ndarray,
                        preamble_wave: np.ndarray) -> Tuple[int, float, float]:
    """Return (peak_index, peak_value, peak_to_mean_ratio).

    Peak-to-mean ratio is a rough confidence score: high values mean the
    correlation is dominated by the preamble match; low values (< ~5)
    suggest the detection may be spurious.
    """
    corr = np.correlate(recording, preamble_wave, mode='valid')
    if len(corr) == 0:
        return 0, 0.0, 0.0
    idx = int(np.argmax(np.abs(corr)))
    peak = float(corr[idx])
    mean_abs = float(np.mean(np.abs(corr))) or 1e-9
    return idx, peak, peak / mean_abs

def demodulate_bit(segment: np.ndarray) -> Tuple[str, float]:
    """Decide '1' or '0' for one coded bit; also return mark-vs-space margin."""
    t = np.linspace(0, BIT_DURATION, len(segment), endpoint=False)
    window = np.hanning(len(t))
    mark_ref  = np.sin(2 * np.pi * F_MARK  * t) * window
    space_ref = np.sin(2 * np.pi * F_SPACE * t) * window
    energy_mark  = np.dot(segment, mark_ref)
    energy_space = np.dot(segment, space_ref)
    bit = '1' if energy_mark > energy_space else '0'
    margin = float(energy_mark - energy_space)
    return bit, margin

def majority_vote(bits: List[str]) -> str:
    return '1' if bits.count('1') > bits.count('0') else '0'

def demodulate_coded_bits(payload: np.ndarray,
                          n_message_bits: int,
                          samples_per_bit: int,
                          label: str = "") -> Tuple[str, int, str]:
    """Demodulate n_message_bits x N_REPEAT coded bits from the start of `payload`.

    Returns (decoded_bits, samples_consumed, raw_repeated_bits_string).
    The raw string is useful for debugging -- you can see whether the
    three copies of each bit agreed or disagreed.
    """
    n_coded = n_message_bits * N_REPEAT
    needed_samples = n_coded * samples_per_bit
    if len(payload) < needed_samples:
        raise ValueError(
            f"Not enough samples for {label}: need {needed_samples}, have {len(payload)}."
        )

    repeated_bits: List[str] = []
    for i in range(n_coded):
        seg = payload[i * samples_per_bit : (i + 1) * samples_per_bit]
        b, _ = demodulate_bit(seg)
        repeated_bits.append(b)

    decoded: List[str] = []
    for i in range(0, n_coded, N_REPEAT):
        decoded.append(majority_vote(repeated_bits[i : i + N_REPEAT]))

    return ''.join(decoded), needed_samples, ''.join(repeated_bits)

def dbg(msg: str) -> None:
    if DEBUG:
        print(msg)

# ---------------------------------------------------------------------------
# Payload decoding
# ---------------------------------------------------------------------------
def bits_to_text(bits: str) -> str:
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break
        try:
            chars.append(chr(int(byte, 2)))
        except ValueError:
            chars.append('?')
    return ''.join(chars)

# ---------------------------------------------------------------------------
# Top-level receive
# ---------------------------------------------------------------------------
def receive_and_decode() -> Optional[str]:
    recording = record_audio(RECORD_DURATION)
    preamble_wave = build_preamble_wave()

    start_idx, peak, pmr = find_preamble_start(recording, preamble_wave)
    print(f"[rx] preamble @ sample {start_idx} "
          f"(peak={peak:.1f}, peak/mean={pmr:.1f})")
    if pmr < 5:
        print("[rx] WARNING: low peak-to-mean ratio -- preamble detection may be spurious.")

    cursor = start_idx + len(preamble_wave)
    samples_per_bit = int(FS * BIT_DURATION)

    # --- Length header + inverted copy -------------------------------------
    try:
        length_str, consumed, raw_len = demodulate_coded_bits(
            recording[cursor:], LENGTH_BITS, samples_per_bit, label="length"
        )
        cursor += consumed
        length_inv_str, consumed, raw_len_inv = demodulate_coded_bits(
            recording[cursor:], LENGTH_BITS, samples_per_bit, label="length_inv"
        )
        cursor += consumed
    except ValueError as e:
        print(f"[rx] header decode failed: {e}")
        return None

    length_val     = int(length_str, 2)
    length_inv_val = int(length_inv_str, 2)
    expected_inv   = (~length_val) & LENGTH_MASK

    dbg(f"[rx] raw length bits:     {raw_len}")
    dbg(f"[rx] raw length_inv bits: {raw_len_inv}")
    dbg(f"[rx] length     = {length_val} (0x{length_val:04X})")
    dbg(f"[rx] length_inv = {length_inv_val} (0x{length_inv_val:04X}) "
        f"expected 0x{expected_inv:04X}")

    if length_inv_val != expected_inv:
        bit_disagreements = bin(length_inv_val ^ expected_inv).count('1')
        print(
            f"[rx] HEADER CORRUPTED: length and length_inv disagree "
            f"in {bit_disagreements} bit(s). Frame rejected."
        )
        return None

    # --- Sanity bound on payload length ------------------------------------
    remaining_samples = len(recording) - cursor
    needed_samples = (length_val + CRC_BITS) * N_REPEAT * samples_per_bit
    if length_val <= 0 or needed_samples > remaining_samples:
        print(
            f"[rx] length ({length_val} bits) doesn't fit in recording. "
            f"Increase RECORD_DURATION."
        )
        return None
    print(f"[rx] length OK: {length_val} payload bits")

    # --- Payload -----------------------------------------------------------
    payload_bits, consumed, _ = demodulate_coded_bits(
        recording[cursor:], length_val, samples_per_bit, label="payload"
    )
    cursor += consumed

    # --- Payload CRC -------------------------------------------------------
    crc_bits, _, _ = demodulate_coded_bits(
        recording[cursor:], CRC_BITS, samples_per_bit, label="crc"
    )
    received_crc = int(crc_bits, 2)
    expected_crc = crc16_ccitt(payload_bits)

    message = bits_to_text(payload_bits)

    if received_crc == expected_crc:
        print(f"[rx] CRC OK (0x{received_crc:04X})")
        return message
    else:
        print(
            f"[rx] CRC MISMATCH (got 0x{received_crc:04X}, "
            f"expected 0x{expected_crc:04X}). Payload may be corrupted."
        )
        return message   # still return what we decoded; flip to None to suppress

if __name__ == "__main__":
    msg = receive_and_decode()
    print(f"Received message: {msg!r}")