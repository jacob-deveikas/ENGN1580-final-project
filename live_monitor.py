from __future__ import annotations

import argparse
import queue
import sys

import numpy as np

from graded_common import FS


def parse_args():
    p = argparse.ArgumentParser(description="Live receiver instrumentation: waveform and 20 Hz-20 kHz spectrum.")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--seconds", type=float, default=0.25)
    p.add_argument("--ylim", type=float, default=1.0)
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)
    hist = np.zeros(int(round(args.seconds*FS)), dtype=np.float32)

    def cb(indata, frames, time, status):
        if status:
            print(f"[monitor] {status}", file=sys.stderr)
        try:
            q.put_nowait(indata[:,0].copy())
        except queue.Full:
            pass

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,7))
    t = np.arange(len(hist))/FS
    line1, = ax1.plot(t, hist, lw=0.8)
    ax1.set_title("Real-time microphone waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-args.ylim, args.ylim)
    ax1.grid(True, alpha=0.3)
    freqs = np.fft.rfftfreq(len(hist), 1/FS)
    line2, = ax2.plot(freqs, np.zeros_like(freqs), lw=0.8)
    ax2.set_xlim(20, 20000)
    ax2.set_ylim(-120, 10)
    ax2.set_title("Real-time power spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    def update(_):
        nonlocal hist
        while not q.empty():
            block = q.get_nowait()
            hist = np.roll(hist, -len(block))
            hist[-len(block):] = block
        line1.set_ydata(hist)
        win = np.hanning(len(hist))
        spec = np.abs(np.fft.rfft(hist*win))**2
        db = 10*np.log10(spec + 1e-12)
        line2.set_ydata(db)
        return line1, line2

    with sd.InputStream(samplerate=FS, channels=1, dtype="float32", blocksize=args.block, device=args.device, callback=cb):
        anim = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
        fig._live_monitor_animation = anim
        plt.show(block=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
