import numpy as np
import sounddevice as sd

# Must match transmitter
fs = 44100
bit_duration = 0.02
f_carrier = 2000
amplitude = 0.5

PREAMBLE = "1010101010101010"

def generate_template(bit):
    t = np.linspace(0, bit_duration, int(fs * bit_duration), endpoint=False)
    carrier = amplitude * np.sin(2 * np.pi * f_carrier * t)
    return carrier if bit == '1' else np.zeros_like(carrier)

def build_preamble_signal():
    return np.concatenate([generate_template(b) for b in PREAMBLE])

def matched_filter(rx):
    N = int(fs * bit_duration)
    t = np.linspace(0, bit_duration, N, endpoint=False)
    ref_sin = np.sin(2 * np.pi * f_carrier * t)
    ref_cos = np.cos(2 * np.pi * f_carrier * t)
    I = np.convolve(rx, ref_sin[::-1], mode='valid')
    Q = np.convolve(rx, ref_cos[::-1], mode='valid')
    mf_mag = np.sqrt(I**2 + Q**2)
    return mf_mag

def record_signal(duration):
    print("Recording...")
    x = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return x.flatten()

def find_preamble_start(rx, preamble_signal):
    corr = np.correlate(rx, preamble_signal, mode='valid')
    start_idx = np.argmax(corr)
    return start_idx

def decode_bits(mf_mag, start_idx):
    N = int(fs * bit_duration)
    bit_values = []

    i = start_idx
    while i + N <= len(mf_mag):
        window = mf_mag[i:i + N]
        bit_values.append(np.max(window))
        i += N

    bit_values = np.array(bit_values)
    high = np.percentile(bit_values, 90)
    low = np.percentile(bit_values, 10)
    threshold = 0.5 * (high + low)
    bits = ['1' if v > threshold else '0' for v in bit_values]
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

    mf_mag = matched_filter(rx)
    N = int(fs * bit_duration)
    payload_start = start + len(PREAMBLE) * N
    bits = decode_bits(mf_mag, payload_start)

    text = bits_to_text(bits)
    print("Decoded text:", text)