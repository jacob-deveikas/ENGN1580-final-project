"""
Improved audio data receiver for laptop‑to‑laptop communication.

This receiver complements ``transmitter_improved.py`` and is designed
to demodulate a BFSK signal containing a Barker preamble and a
triple‑repetition coded payload.  The receiver performs the
following steps:

1. Record a segment of audio from the computer's microphone.
2. Generate a template of the preamble waveform using the same
   parameters as the transmitter (Barker sequence, BFSK frequencies,
   pulse shaping).  Cross‑correlate this template with the recorded
   audio to locate the start of the transmitted frame.  Barker
   sequences have near‑ideal autocorrelation properties, making
   detection reliable even in noise【853951921992884†L148-L171】.
3. Once aligned, process each bit duration by computing the
   correlation of the signal with both mark and space tones.  A
   decision for each repeated bit is made by comparing the energies of
   the two tones.  Groups of three repeated bits are then
   majority‑voted to reconstruct the original message bit.
4. Reassemble the bitstream into bytes and convert to ASCII text.

Matched filtering and correlation demodulators are optimal for
detecting known signals in additive white Gaussian noise【485443521237996†L343-L347】.  In
this implementation, a simple inner‑product (correlation) with the
appropriate tone serves as an approximation to a matched filter.

Note: The current implementation records for a fixed duration.  If
your messages are longer, increase ``RECORD_DURATION`` accordingly.
"""

import numpy as np
import sounddevice as sd
from typing import List

# Import parameters from the transmitter.  If these constants are
# changed in the transmitter, they must be updated here to match.
from transmitter_improved import (
    FS,
    BIT_DURATION,
    F_MARK,
    F_SPACE,
    AMPLITUDE,
    PREAMBLE_BITS,
)

# Number of times each bit was repeated in the transmitter
N_REPEAT = 3

# Duration of the recording (seconds).  It should be long enough to
# capture the entire transmitted frame.  Adjust based on message length.
RECORD_DURATION = 6.0

def record_audio(duration: float = RECORD_DURATION) -> np.ndarray:
    """Record audio from the default input device for ``duration`` seconds.

    Args:
        duration: Recording length in seconds.
    Returns:
        A NumPy array containing the recorded audio samples.
    """
    print(f"Recording for {duration} s …")
    recording = sd.rec(int(duration * FS), samplerate=FS, channels=1, blocking=True)
    return recording.flatten()

def generate_bit_wave(bit: str) -> np.ndarray:
    """Generate the BFSK waveform for a single bit (used for preamble template).

    This mirrors ``generate_bit_wave()`` from the transmitter.  It is
    intentionally duplicated here so the receiver does not need to
    import all of the transmitter functions.

    Args:
        bit: '1' or '0'.
    Returns:
        A 1‑D NumPy array containing the waveform samples.
    """
    t = np.linspace(0, BIT_DURATION, int(FS * BIT_DURATION), endpoint=False)
    freq = F_MARK if bit == '1' else F_SPACE
    tone = np.sin(2 * np.pi * freq * t)
    window = np.hanning(len(t))
    return AMPLITUDE * tone * window

def build_preamble_wave() -> np.ndarray:
    """Construct the concatenated waveform for the Barker preamble."""
    segments = [generate_bit_wave(b) for b in PREAMBLE_BITS]
    return np.concatenate(segments)

def find_preamble_start(recording: np.ndarray, preamble_wave: np.ndarray) -> int:
    """Locate the start of the transmitted frame by cross‑correlation.

    Args:
        recording: The recorded audio samples.
        preamble_wave: The known BFSK waveform of the preamble.
    Returns:
        The index into ``recording`` where the preamble begins.  If
        correlation fails, returns 0.
    """
    # Compute the cross‑correlation between the recording and the
    # preamble template.  We use 'valid' mode to ensure the sliding
    # template fully overlaps the recording segment.  The index of the
    # maximum correlation is our estimated start position.
    corr = np.correlate(recording, preamble_wave, mode='valid')
    if len(corr) == 0:
        return 0
    start_idx = int(np.argmax(corr))
    return start_idx

def demodulate_bit(segment: np.ndarray) -> str:
    """Demodulate a single repeated bit using correlation with mark/space tones.

    Args:
        segment: A 1‑D array of samples corresponding to one bit duration.
    Returns:
        '1' if the mark frequency energy exceeds the space frequency energy,
        otherwise '0'.
    """
    # Time vector for the segment
    t = np.linspace(0, BIT_DURATION, len(segment), endpoint=False)
    # Compute correlations (inner products) with the mark and space tones.
    mark_ref = np.sin(2 * np.pi * F_MARK * t) * np.hanning(len(t))
    space_ref = np.sin(2 * np.pi * F_SPACE * t) * np.hanning(len(t))
    energy_mark = np.dot(segment, mark_ref)
    energy_space = np.dot(segment, space_ref)
    return '1' if energy_mark > energy_space else '0'

def majority_vote(bits: List[str]) -> str:
    """Given a list of bits (strings), return the bit that appears most.

    Assumes the list length equals ``N_REPEAT``.  Ties default to '0'.
    """
    ones = bits.count('1')
    zeros = bits.count('0')
    return '1' if ones > zeros else '0'

def decode_payload(bits: str) -> str:
    """Convert a binary string into ASCII text.

    Args:
        bits: The bit string where each byte corresponds to one character.
    Returns:
        The decoded ASCII message.
    """
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

def receive_and_decode() -> str:
    """Record audio, detect and demodulate the BFSK frame, and return the message."""
    # Record audio from microphone
    recording = record_audio(RECORD_DURATION)
    # Build the preamble waveform
    preamble_wave = build_preamble_wave()
    # Find the start of the preamble in the recording
    start_idx = find_preamble_start(recording, preamble_wave)
    print(f"Estimated frame start index: {start_idx}")
    # Skip the preamble waveform to get to the payload start
    payload_start = start_idx + len(preamble_wave)
    # Compute number of samples per bit
    samples_per_bit = int(FS * BIT_DURATION)
    # Extract the payload portion of the recording
    payload = recording[payload_start:]
    # Determine how many repeated bits we can demodulate from the payload
    n_repeated_bits = len(payload) // samples_per_bit
    # Demodulate each repeated bit
    repeated_bits: List[str] = []
    for i in range(n_repeated_bits):
        segment = payload[i * samples_per_bit:(i + 1) * samples_per_bit]
        repeated_bits.append(demodulate_bit(segment))
    # Group repeated bits and perform majority voting
    decoded_bits = []
    for i in range(0, len(repeated_bits), N_REPEAT):
        group = repeated_bits[i:i + N_REPEAT]
        if len(group) < N_REPEAT:
            break
        decoded_bits.append(majority_vote(group))
    bit_str = ''.join(decoded_bits)
    # Convert bit string to ASCII text
    message = decode_payload(bit_str)
    return message

if __name__ == "__main__":
    msg = receive_and_decode()
    print(f"Received message: {msg}")
