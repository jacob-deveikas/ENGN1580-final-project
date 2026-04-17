import numpy as np
import sounddevice as sd

# Parameters (tune these)
fs = 44100              # Sampling rate (Hz)
bit_duration = 0.02      # seconds per bit
f_carrier = 2000        # carrier frequency (Hz)
amplitude = 1         # signal amplitude
PREAMBLE = "1010101010101010"   # alternating pattern

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def generate_ook_signal(bitstream):
    t = np.linspace(0, bit_duration, int(fs * bit_duration), endpoint=False)
    carrier = amplitude * np.sin(2 * np.pi * f_carrier * t)

    signal = []

    for bit in bitstream:
        if bit == '1':
            signal.append(carrier)
        else:
            signal.append(np.zeros_like(carrier))

    return np.concatenate(signal)

def transmit(text):
    bits = text_to_bits(text)
    print("Transmitting bits:", bits)
    frame = PREAMBLE + bits

    signal = generate_ook_signal(frame)

    sd.play(signal, fs)
    sd.wait()

if __name__ == "__main__":
    transmit("WAKE UP, NEO...")
