"""
Improved audio data transmitter for laptop-to-laptop communication.

Frame structure (bit order, top to bottom):

    [ Barker-13 preamble      ] 13 bits, NOT repeated (used as correlation template)
    [ Length header           ] 16 bits x 3 repetition = 48 coded bits
    [ Payload                 ] N  bits x 3 repetition = 3N coded bits
    [ CRC-16-CCITT of payload ] 16 bits x 3 repetition = 48 coded bits

The length header tells the receiver exactly how many payload bits follow,
so the receiver stops reading at the correct boundary instead of
demodulating trailing microphone noise. The CRC is a final integrity
check so corrupted frames can be detected instead of silently accepted.
"""

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Signalling parameters
# ---------------------------------------------------------------------------
FS = 44100            # sampling rate in Hz
BIT_DURATION = 0.02   # duration of one repeated (coded) bit, seconds
F_MARK = 2000         # frequency for bit 1
F_SPACE = 3000        # frequency for bit 0
AMPLITUDE = 0.5       # signal amplitude (< 1 to avoid clipping)

# Barker-13 for preamble synchronisation
BARKER_13 = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
PREAMBLE_BITS = ''.join('1' if x > 0 else '0' for x in BARKER_13)

# Framing parameters
N_REPEAT = 3          # repetition coding factor for length / payload / CRC
LENGTH_BITS = 16      # width of the length header (supports up to 65535 payload bits)
CRC_BITS = 16         # CRC-16-CCITT

# ---------------------------------------------------------------------------
# Bit / byte helpers
# ---------------------------------------------------------------------------
def text_to_bits(text: str) -> str:
    """Convert an ASCII string to its 8-bits-per-char binary representation."""
    return ''.join(format(ord(c), '08b') for c in text)

def repeat_bits(bits: str, n_repeat: int = N_REPEAT) -> str:
    """Repeat each bit n_repeat times (simple (n,1) repetition code)."""
    return ''.join(bit * n_repeat for bit in bits)

def crc16_ccitt(bits: str) -> int:
    """Bit-serial CRC-16-CCITT (poly 0x1021, init 0xFFFF).

    Operates directly on a bit string so the CRC covers exactly the
    payload bits the receiver will recover, independent of byte alignment.
    """
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
    """BFSK waveform for one coded bit, windowed with a Hann envelope."""
    t = np.linspace(0, BIT_DURATION, int(FS * BIT_DURATION), endpoint=False)
    freq = F_MARK if bit == '1' else F_SPACE
    tone = np.sin(2 * np.pi * freq * t)
    window = np.hanning(len(t))
    return AMPLITUDE * tone * window

def assemble_signal(bitstream: str) -> np.ndarray:
    """Concatenate per-bit waveforms into a single audio buffer."""
    segments = [generate_bit_wave(b) for b in bitstream]
    return np.concatenate(segments)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_frame_bits(text: str) -> str:
    """Build the full coded bitstream (preamble + length + payload + CRC)."""
    payload_bits = text_to_bits(text)
    n_payload = len(payload_bits)
    if n_payload >= (1 << LENGTH_BITS):
        raise ValueError(
            f"Payload too long: {n_payload} bits exceeds {LENGTH_BITS}-bit length field."
        )

    length_bits = format(n_payload, f'0{LENGTH_BITS}b')
    crc_value = crc16_ccitt(payload_bits)
    crc_bits = format(crc_value, f'0{CRC_BITS}b')

    coded_length  = repeat_bits(length_bits)
    coded_payload = repeat_bits(payload_bits)
    coded_crc     = repeat_bits(crc_bits)

    return PREAMBLE_BITS + coded_length + coded_payload + coded_crc

def transmit(text: str) -> None:
    """Encode and play an ASCII message over the default audio output."""
    if not text:
        raise ValueError("Cannot transmit an empty message.")

    frame_bits = build_frame_bits(text)
    payload_bits = text_to_bits(text)

    print(
        f"[tx] payload={len(payload_bits)} bits, "
        f"crc=0x{crc16_ccitt(payload_bits):04X}, "
        f"total coded bits={len(frame_bits)}, "
        f"duration={len(frame_bits) * BIT_DURATION:.2f}s"
    )

    signal = assemble_signal(frame_bits)
    sd.play(signal, FS)
    sd.wait()

if __name__ == "__main__":
    transmit("HELLO")
