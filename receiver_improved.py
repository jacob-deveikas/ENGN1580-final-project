"""
Improved audio data receiver for laptop-to-laptop communication.

Matches the framing in transmitter_improved.py:

    [ Barker-13 preamble      ] -> located via cross-correlation
    [ Length header: 16 bits  ] x3 repetition -> tells us payload length
    [ Payload: N bits         ] x3 repetition -> read exactly N bits
    [ CRC-16-CCITT: 16 bits   ] x3 repetition -> verify integrity

Using the length prefix means we stop reading at a deterministic sample
offset; trailing microphone noise can no longer bleed into the decoded
text. The CRC flags corrupted frames instead of quietly returning
garbage.
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
    CRC_BITS,
    crc16_ccitt,
)

RECORD_DURATION = 6.0  # seconds; increase for longer messages

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
def record_audio(duration: float = RECORD_DURATION) -> np.ndarray:
    print(f"[rx] Recording for {duration} s ...")
    recording = sd.rec(int(duration * FS), samplerate=FS, channels=1, blocking=True)
    return recording.flatten()

# ---------------------------------------------------------------------------
# Waveform templates (must match transmitter)
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
def find_preamble_start(recording: np.ndarray, preamble_wave: np.ndarray) -> int:
    corr = np.correlate(recording, preamble_wave, mode='valid')
    if len(corr) == 0:
        return 0
    return int(np.argmax(corr))

def demodulate_bit(segment: np.ndarray) -> str:
    """Correlate a one-bit-duration segment against mark and space tones."""
    t = np.linspace(0, BIT_DURATION, len(segment), endpoint=False)
    window = np.hanning(len(t))
    mark_ref  = np.sin(2 * np.pi * F_MARK  * t) * window
    space_ref = np.sin(2 * np.pi * F_SPACE * t) * window
    energy_mark  = np.dot(segment, mark_ref)
    energy_space = np.dot(segment, space_ref)
    return '1' if energy_mark > energy_space else '0'

def majority_vote(bits: List[str]) -> str:
    return '1' if bits.count('1') > bits.count('0') else '0'

def demodulate_coded_bits(payload: np.ndarray,
                          n_message_bits: int,
                          samples_per_bit: int) -> Tuple[str, int]:
    """Demodulate n_message_bits information bits (each sent x N_REPEAT times).

    Returns:
        (decoded_bits, samples_consumed)
    """
    n_coded = n_message_bits * N_REPEAT
    needed_samples = n_coded * samples_per_bit
    if len(payload) < needed_samples:
        raise ValueError(
            f"Recording too short: need {needed_samples} samples, have {len(payload)}."
        )

    repeated_bits: List[str] = []
    for i in range(n_coded):
        seg = payload[i * samples_per_bit : (i + 1) * samples_per_bit]
        repeated_bits.append(demodulate_bit(seg))

    decoded: List[str] = []
    for i in range(0, n_coded, N_REPEAT):
        decoded.append(majority_vote(repeated_bits[i : i + N_REPEAT]))

    return ''.join(decoded), needed_samples

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

    start_idx = find_preamble_start(recording, preamble_wave)
    print(f"[rx] preamble detected at sample {start_idx}")

    # Cursor walks through the recording one field at a time.
    cursor = start_idx + len(preamble_wave)
    samples_per_bit = int(FS * BIT_DURATION)

    # --- Length header -----------------------------------------------------
    try:
        length_str, consumed = demodulate_coded_bits(
            recording[cursor:], LENGTH_BITS, samples_per_bit
        )
    except ValueError as e:
        print(f"[rx] length decode failed: {e}")
        return None
    cursor += consumed
    payload_len = int(length_str, 2)
    print(f"[rx] length header = {payload_len} payload bits")

    # Sanity bound: payload must fit inside the remaining recording, with
    # room for the CRC trailer.
    remaining_samples = len(recording) - cursor
    needed_samples = (payload_len + CRC_BITS) * N_REPEAT * samples_per_bit
    if payload_len <= 0 or needed_samples > remaining_samples:
        print(
            f"[rx] implausible length ({payload_len} bits). "
            f"Header likely corrupted. Aborting."
        )
        return None

    # --- Payload -----------------------------------------------------------
    payload_bits, consumed = demodulate_coded_bits(
        recording[cursor:], payload_len, samples_per_bit
    )
    cursor += consumed

    # --- CRC trailer -------------------------------------------------------
    crc_bits, _ = demodulate_coded_bits(
        recording[cursor:], CRC_BITS, samples_per_bit
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
        # Return the message anyway so you can see what was decoded;
        # flip to `return None` if you'd rather suppress bad frames.
        return message

if __name__ == "__main__":
    msg = receive_and_decode()
    print(f"Received message: {msg!r}")