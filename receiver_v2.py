import numpy as np
import sounddevice as sd

# Must match transmitter
fs = 44100
bit_duration = 0.1
f_carrier = 2000
amplitude = 0.5

PREAMBLE = "1010101010101010"

def generate_template(bit):
    t = np.linspace(0, bit_duration, int(fs * bit_duration), endpoint=False)
    carrier = amplitude * np.sin(2 * np.pi * f_carrier * t)
    return carrier if bit == '1' else np.zeros_like(carrier)

def build_preamble_signal():
    return np.concatenate([generate_template(b) for b in PREAMBLE])

def record_signal(duration):
    print("Recording...")
    x = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return x.flatten()

def find_preamble_start(rx, preamble_signal):
    corr = np.correlate(rx, preamble_signal, mode='valid')
    start_idx = np.argmax(corr)
    return start_idx

def decode_bits(rx, start_idx):
    N = int(fs * bit_duration)
    bits = []

    i = start_idx
    while i + N <= len(rx):
        chunk = rx[i:i+N]

        # Energy detection
        energy = np.sum(chunk**2)

        # Simple threshold (can be improved)
        bits.append('1' if energy > 0.01 else '0')

        i += N

    return ''.join(bits)

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

if __name__ == "__main__":
    duration = 5  # seconds (must exceed transmission time)
    rx = record_signal(duration)

    preamble_signal = build_preamble_signal()
    start = find_preamble_start(rx, preamble_signal)

    print("Preamble detected at index:", start)

    bits = decode_bits(rx, start + len(preamble_signal))

    text = bits_to_text(bits)
    print("Decoded text:", text)