from __future__ import annotations

import argparse
import queue
import sys

import numpy as np

from graded_common import FS, qpsk_params


def parse_args():
    p = argparse.ArgumentParser(description="Live receiver-side QPSK constellation monitor. Instrumentation only, not the Pe decoder.")
    p.add_argument("--carrier", type=float, required=True)
    p.add_argument("--rate", type=float, default=500.0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--block", type=int, default=2048)
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

    sps, actual_rate, _ = qpsk_params(args.rate)
    n_hist = int(round(args.seconds * FS))
    hist = np.zeros(n_hist, dtype=np.float32)
    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)

    def cb(indata, frames, time, status):
        if status:
            print(f"[qpsk-live] {status}", file=sys.stderr)
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=16, alpha=0.65)
    ideal = np.array([(1+1j)/np.sqrt(2), (-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2), (1-1j)/np.sqrt(2)])
    ax.scatter(np.real(ideal), np.imag(ideal), marker="x", s=100, label="ideal")
    ax.axhline(0, lw=0.8)
    ax.axvline(0, lw=0.8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Live QPSK constellation, fc={args.carrier:.0f} Hz, rate={actual_rate:.1f} bps")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")

    def update(_):
        nonlocal hist
        while not q.empty():
            block = q.get_nowait()
            hist = np.roll(hist, -len(block))
            hist[-len(block):] = block
        n = np.arange(len(hist), dtype=np.float64)
        bb = 2.0 * hist * np.exp(-1j * 2 * np.pi * args.carrier * n / FS)
        # Try a few phases and choose the one with largest average symbol magnitude.
        best = None
        best_pow = -1.0
        for off in range(0, min(sps, 80), max(1, sps // 40)):
            usable = (len(bb) - off) // sps
            if usable < 4:
                continue
            seg = bb[off:off + usable * sps].reshape(usable, sps).mean(axis=1)
            pwr = float(np.mean(np.abs(seg)))
            if pwr > best_pow:
                best_pow = pwr
                best = seg[-250:]
        if best is not None and len(best):
            # Normalize just for display.
            pts = best / (np.median(np.abs(best)) + 1e-9) / np.sqrt(2)
            scat.set_offsets(np.column_stack([np.real(pts), np.imag(pts)]))
        return (scat,)

    with sd.InputStream(samplerate=FS, channels=1, dtype="float32", blocksize=args.block, device=args.device, callback=cb):
        anim = animation.FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)
        fig._qpsk_anim = anim
        plt.show(block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
