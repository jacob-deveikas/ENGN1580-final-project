"""
Improved audio data transmitter for laptop-to-laptop communication.

Frame structure:

    [ Barker-13 preamble            ] 13 bits, NOT repeated
    [ Length header                 ] 16 bits x 3 repetition
    [ Length header (bitwise NOT)   ] 16 bits x 3 repetition   <-- header self-check
    [ Payload                       ] N  bits x 3 repetition
    [ CRC-16-CCITT of payload       ] 16 bits x 3 repetition

Why the inverted length copy:
  The length header is a single point of failure -- if it's wrong, the
  receiver slices the wrong number of samples and everything downstream
  is garbage. Triple repetition alone doesn't protect against
  correlated bit errors (e.g., a tone notch in the room response).
  Sending the length twice, with the second copy bitwise-inverted,
  means the receiver can verify `length XOR length_inv == 0xFFFF`
  and reject the frame cleanly when the header is corrupt.
"""

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Signalling parameters
# ---------------------------------------------------------------------------
FS = 44100
BIT_DURATION = 0.02
F_MARK = 2000
F_SPACE = 3000
AMPLITUDE = 0.5

BARKER_13 = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
PREAMBLE_BITS = ''.join('1' if x > 0 else '0' for x in BARKER_13)

# Framing parameters
N_REPEAT = 3
LENGTH_BITS = 16
CRC_BITS = 16
LENGTH_MASK = (1 << LENGTH_BITS) - 1   # 0xFFFF

# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------
def text_to_bits(text: str) -> str:
    return ''.join(format(ord(c), '08b') for c in text)

def repeat_bits(bits: str, n_repeat: int = N_REPEAT) -> str:
    return ''.join(bit * n_repeat for bit in bits)

def crc16_ccitt(bits: str) -> int:
    crc = 0xFFFF
    for bit_char in bits:
        bit = 1 if bit_char == '1' else 0
        top = (crc >> 15) & 1
        crc = (crc << 1) & 0xFFFF
        if top ^ bit:
            crc ^= 0x1021
    return crc

# ---------------------------------------------------------------------------
# Waveform generation
# ---------------------------------------------------------------------------
def generate_bit_wave(bit: str) -> np.ndarray:
    t = np.linspace(0, BIT_DURATION, int(FS * BIT_DURATION), endpoint=False)
    freq = F_MARK if bit == '1' else F_SPACE
    tone = np.sin(2 * np.pi * freq * t)
    window = np.hanning(len(t))
    return AMPLITUDE * tone * window

def assemble_signal(bitstream: str) -> np.ndarray:
    return np.concatenate([generate_bit_wave(b) for b in bitstream])

# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------
def build_frame_bits(text: str) -> str:
    payload_bits = text_to_bits(text)
    n_payload = len(payload_bits)
    if n_payload >= (1 << LENGTH_BITS):
        raise ValueError(
            f"Payload too long: {n_payload} bits exceeds {LENGTH_BITS}-bit length field."
        )

    length_value     = n_payload
    length_inv_value = (~length_value) & LENGTH_MASK

    length_bits     = format(length_value,     f'0{LENGTH_BITS}b')
    length_inv_bits = format(length_inv_value, f'0{LENGTH_BITS}b')

    crc_value = crc16_ccitt(payload_bits)
    crc_bits  = format(crc_value, f'0{CRC_BITS}b')

    coded_length     = repeat_bits(length_bits)
    coded_length_inv = repeat_bits(length_inv_bits)
    coded_payload    = repeat_bits(payload_bits)
    coded_crc        = repeat_bits(crc_bits)

    return PREAMBLE_BITS + coded_length + coded_length_inv + coded_payload + coded_crc

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def transmit(text: str) -> None:
    if not text:
        raise ValueError("Cannot transmit an empty message.")

    frame_bits = build_frame_bits(text)
    payload_bits = text_to_bits(text)

    print(
        f"[tx] payload={len(payload_bits)} bits "
        f"(len=0x{len(payload_bits):04X}, "
        f"len_inv=0x{(~len(payload_bits)) & LENGTH_MASK:04X}), "
        f"crc=0x{crc16_ccitt(payload_bits):04X}, "
        f"total coded bits={len(frame_bits)}, "
        f"duration={len(frame_bits) * BIT_DURATION:.2f}s"
    )

    signal = assemble_signal(frame_bits)
    sd.play(signal, FS)
    sd.wait()

if __name__ == "__main__":
    transmit("HELLO")