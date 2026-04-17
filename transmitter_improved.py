"""
Improved audio data transmitter for laptop‑to‑laptop communication.

This module implements a more robust signalling scheme than the
baseline OOK example.  It uses binary frequency‑shift keying (BFSK)
with pulse shaping and a Barker code preamble to achieve better
synchronisation and noise resilience.  To further protect against
errors, each information bit is repeated three times (a simple
repetition code) so that the receiver can perform majority‑vote
decoding.  The resulting waveform is played through the computer
speaker via the sounddevice library.

Key design choices:

* BFSK modulation: A binary 1 is represented by a tone of frequency
  ``F_MARK`` and a binary 0 by ``F_SPACE``.  This modulation has
  constant amplitude, which makes it less sensitive to amplitude noise
  and easier to detect reliably【588465767416327†L256-L264】.

* Pulse shaping: Each bit is multiplied by a Hann window so that
  transitions between tones do not introduce large spectral
  side‑bands.  Pulse shaping confines the signal’s bandwidth and
  reduces the chance of spectral splatter【950297419617221†L220-L231】.

* Preamble: A Barker sequence is used as a preamble.  Barker codes
  possess near‑ideal autocorrelation properties and are widely used
  as synchronising patterns in communication systems【853951921992884†L148-L171】.  At the
  receiver, cross‑correlation with the BFSK‑modulated preamble
  provides an accurate estimate of the start of the message.

* Simple error protection: Each message bit is repeated three
  times (a (3,1) repetition code).  At the receiver, the three
  demodulated bits are combined using majority vote.  This technique
  corrects single‑bit errors within each trio and improves
  robustness without requiring a complex error‑correcting code.

To send a message, call ``transmit(<string>)``.  The transmitter
converts characters into a stream of bits, adds the Barker preamble
and repetition coding, generates the BFSK waveform and plays it.
"""

import numpy as np
import sounddevice as sd

# Sampling parameters
FS = 44100  # sampling rate in Hz
BIT_DURATION = 0.02  # bit duration in seconds (20 ms per repeated bit)

# BFSK frequencies in Hz.  These values are chosen to fit within
# the passband of typical laptop speakers while being sufficiently
# separated to simplify detection.  If you characterise the
# frequency response of your specific hardware, you may wish to
# adjust these values accordingly.
F_MARK = 2000  # frequency representing bit 1
F_SPACE = 3000  # frequency representing bit 0

# Amplitude of the transmitted signal.  Must be less than 1 to
# avoid clipping.  Lower amplitudes reduce audible annoyance and
# distortion.
AMPLITUDE = 0.5

# Barker‑13 code (values +1/−1) for robust synchronisation.  When
# mapped to bits, +1→1 and −1→0.  See Wikipedia for Barker
# sequences【853951921992884†L148-L171】.
BARKER_13 = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]

# Convert ±1 Barker chips to binary preamble string
PREAMBLE_BITS = ''.join('1' if x > 0 else '0' for x in BARKER_13)

def text_to_bits(text: str) -> str:
    """Convert a string into its binary representation (8 bits per char)."""
    return ''.join(format(ord(c), '08b') for c in text)

def repeat_bits(bits: str, n_repeat: int = 3) -> str:
    """Repeat each bit ``n_repeat`` times for simple redundancy.

    Args:
        bits: The original bit string.
        n_repeat: How many times to repeat each bit.
    Returns:
        A new bit string where each bit appears ``n_repeat`` times consecutively.
    """
    return ''.join(bit * n_repeat for bit in bits)

def generate_bit_wave(bit: str) -> np.ndarray:
    """Generate the BFSK waveform for a single bit with pulse shaping.

    Args:
        bit: '1' or '0'.
    Returns:
        A 1‑D NumPy array containing the waveform samples for the bit.
    """
    # time vector for one bit
    t = np.linspace(0, BIT_DURATION, int(FS * BIT_DURATION), endpoint=False)
    # choose the appropriate frequency
    freq = F_MARK if bit == '1' else F_SPACE
    tone = np.sin(2 * np.pi * freq * t)
    # apply Hann window (smooth on/off) to reduce spectral splatter
    window = np.hanning(len(t))
    return AMPLITUDE * tone * window

def assemble_signal(bitstream: str) -> np.ndarray:
    """Convert an entire bitstream into a concatenated waveform.

    Args:
        bitstream: String of '0' and '1' characters.
    Returns:
        A NumPy array representing the concatenated BFSK waveform.
    """
    segments = [generate_bit_wave(b) for b in bitstream]
    return np.concatenate(segments)

def transmit(text: str) -> None:
    """Transmit text via BFSK audio through the computer's speakers.

    The function prepends a Barker‑coded preamble and applies triple
    redundancy to the payload bits.  The assembled waveform is played
    asynchronously using sounddevice and the function waits for
    completion.

    Args:
        text: The ASCII message to send.
    """
    if not text:
        raise ValueError("Cannot transmit an empty message.")
    # Convert text to bits
    payload_bits = text_to_bits(text)
    # Apply repetition coding for error resilience
    coded_bits = repeat_bits(payload_bits, n_repeat=3)
    # Compose frame: preamble + payload
    frame_bits = PREAMBLE_BITS + coded_bits
    print(f"Transmitting {len(payload_bits)} message bits (with redundancy {len(coded_bits)} bits).")
    # Generate full signal waveform
    signal = assemble_signal(frame_bits)
    # Play the signal
    sd.play(signal, FS)
    sd.wait()

if __name__ == "__main__":
    # Example usage
    transmit("HELLO")