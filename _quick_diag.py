"""5-second mic capture diagnostic.

Records 5 seconds from --device, FFTs, prints top 10 peaks and rms.
Expected during a 1 kHz interference test:
   the largest peak should be at ~1000 Hz, and at much higher amplitude
   than the noise floor.
"""
import sys
import numpy as np
import sounddevice as sd

fs = 44100
device = 0 if len(sys.argv) < 2 else int(sys.argv[1])
duration = 5.0
n = int(duration * fs)

print(f"Recording {duration} s from device {device} at {fs} Hz...")
rec = sd.rec(n, samplerate=fs, channels=1, dtype="float32", device=device, blocking=True)
x = rec[:, 0]

rms = float(np.sqrt(np.mean(x ** 2)))
peak = float(np.max(np.abs(x)))
print(f"RMS = {rms:.5f}")
print(f"Peak amplitude = {peak:.5f}")
print(f"DC offset = {float(np.mean(x)):.5f}")

# FFT
win = np.hanning(len(x))
F = np.abs(np.fft.rfft(x * win))
freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
F_db = 20 * np.log10(F + 1e-9)

# Find the 10 largest peaks above 100 Hz (avoid DC)
mask = freqs > 100
top_idx = np.argsort(F[mask])[-10:][::-1]
freqs_m = freqs[mask]
F_db_m = F_db[mask]

print("\nTop 10 spectral peaks above 100 Hz:")
for i in top_idx:
    print(f"  {freqs_m[i]:7.1f} Hz   {F_db_m[i]:6.1f} dB")

# Specifically check 1000 Hz +/- 50 Hz
mask_1k = (freqs > 950) & (freqs < 1050)
mask_500 = (freqs > 450) & (freqs < 550)
mask_2k = (freqs > 1950) & (freqs < 2050)
print("\nKey-frequency power readings:")
print(f"  500 Hz band  (450-550):  {F_db[mask_500].max():.1f} dB")
print(f"  1000 Hz band (950-1050): {F_db[mask_1k].max():.1f} dB")
print(f"  2000 Hz band (1950-2050): {F_db[mask_2k].max():.1f} dB")

# Verdict
if F_db[mask_1k].max() > F_db[mask_500].max() + 12 and F_db[mask_1k].max() > F_db[mask_2k].max() + 12:
    print("\n>>> VERDICT: clear 1 kHz tone present. Cable works. Continue with demo.")
elif rms < 0.0005:
    print("\n>>> VERDICT: mic is silent. No signal at all. Check Mic Mode and device id.")
else:
    print("\n>>> VERDICT: mic IS receiving sound, but no 1 kHz peak. The signal you're seeing is ambient/junk, not the TX tone.")
    print("    -> Most likely: TX laptop's audio is going to its built-in speakers, not into the cable.")
    print("    -> OR: Cubilux / MillSO is wired wrong.")
