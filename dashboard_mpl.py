"""Matplotlib-based dashboard — drop-in replacement for dashboard.py when
PyQt6 is broken. Uses matplotlib.animation. Slower fps but functionally
equivalent.

Covers all 5 instrumentation rubric items:
  1. Real-time TX waveform
  2. Real-time RX waveform
  3. Real-time spectrum 20 Hz - 20 kHz (log x)
  4. Real-time constellations (4x4 grid for OFDM, single for others)
  5. Real-time running-average Pe (EMA + cumulative)

Run examples (same args as dashboard.py):
  python3 dashboard_mpl.py --modem ofdm --in-device 0           # acoustic RX
  python3 dashboard_mpl.py --modem ofdm --mode-audio tx_only --out-device 0
  python3 dashboard_mpl.py --modem fsk --rate 5000 --in-device 0
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("MacOSX")  # explicit mac backend; avoids Qt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import streaming_engine as se
import ofdm_phy as op
from backchannel import FeedbackReceiver, FeedbackSender, detect_jammer_bins
from prbs15 import BERMeter

DEFAULT_FS = 48000


# ============================================================================
class InterferenceSource:
    def __init__(self, fs: int = DEFAULT_FS):
        self.fs = fs
        self.freq = 3000.0
        self.amp = 0.0
        self.white = False
        self.phase = 0.0
        self.lock = threading.Lock()

    def set_freq(self, hz):  self.freq = max(20.0, min(20000.0, float(hz)))
    def set_amp(self, a):    self.amp  = max(0.0, min(1.0, float(a)))
    def set_white(self, on): self.white = bool(on)

    def next_block(self, n):
        with self.lock:
            f, a, white, phase = self.freq, self.amp, self.white, self.phase
        if a <= 1e-6:
            return np.zeros(n, dtype=np.float32)
        if white:
            return (a * 0.7 * np.random.randn(n)).astype(np.float32)
        inc = 2 * math.pi * f / self.fs
        t = phase + inc * np.arange(n)
        sig = np.sin(t).astype(np.float32)
        with self.lock:
            self.phase = (phase + inc * n) % (2 * math.pi)
        return (a * sig).astype(np.float32)


class AudioEngine:
    def __init__(self, tx_engine, rx_engine, interference, fs, blocksize, in_device, out_device, mode):
        self.tx_engine = tx_engine
        self.rx_engine = rx_engine
        self.interference = interference
        self.fs = fs
        self.blocksize = int(blocksize)
        self.in_device = in_device
        self.out_device = out_device
        self.mode = mode
        self.tx_ring = se.RingBuffer(int(2 * fs))
        self.rx_ring = se.RingBuffer(int(2 * fs))
        self._stream = None
        self._latest_rx = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self.tx_gain = 0.85

    def start(self):
        import sounddevice as sd
        if self.mode == "tx_only":
            self._stream = sd.OutputStream(samplerate=self.fs, blocksize=self.blocksize,
                                            channels=2, dtype="float32",
                                            device=self.out_device, callback=self._cb_tx,
                                            latency="low")
        elif self.mode == "rx_only":
            self._stream = sd.InputStream(samplerate=self.fs, blocksize=self.blocksize,
                                           channels=1, dtype="float32",
                                           device=self.in_device, callback=self._cb_rx,
                                           latency="low")
        else:
            try:
                self._stream = sd.Stream(samplerate=self.fs, blocksize=self.blocksize,
                                          channels=(1, 2), dtype="float32",
                                          device=(self.in_device, self.out_device),
                                          callback=self._cb_duplex, latency="low")
            except Exception as e:
                print(f"[audio] stereo duplex failed: {e}; trying mono out", file=sys.stderr)
                self._stream = sd.Stream(samplerate=self.fs, blocksize=self.blocksize,
                                          channels=(1, 1), dtype="float32",
                                          device=(self.in_device, self.out_device),
                                          callback=self._cb_duplex_mono, latency="low")
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _build_tx(self, n):
        if self.tx_engine is None:
            tx = np.zeros(n, dtype=np.float32)
        else:
            try:
                tx = self.tx_engine.next_block(n).astype(np.float32)
            except Exception as e:
                print(f"[audio] tx err: {e}", file=sys.stderr)
                tx = np.zeros(n, dtype=np.float32)
        out = self.tx_gain * tx + self.interference.next_block(n)
        peak = float(np.max(np.abs(out)) or 0.0)
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32)

    def _cb_tx(self, outdata, frames, t, status):
        block = self._build_tx(frames)
        outdata[:, 0] = block
        if outdata.shape[1] >= 2:
            outdata[:, 1] = block
        self.tx_ring.push(block)

    def _cb_rx(self, indata, frames, t, status):
        rx = indata[:, 0].copy()
        self.rx_ring.push(rx)
        with self._lock:
            self._latest_rx = rx

    def _cb_duplex(self, indata, outdata, frames, t, status):
        block = self._build_tx(frames)
        outdata[:, 0] = block
        if outdata.shape[1] >= 2:
            outdata[:, 1] = block
        self.tx_ring.push(block)
        rx = indata[:, 0].copy()
        self.rx_ring.push(rx)
        with self._lock:
            self._latest_rx = rx

    def _cb_duplex_mono(self, indata, outdata, frames, t, status):
        block = self._build_tx(frames)
        outdata[:, 0] = block
        self.tx_ring.push(block)
        rx = indata[:, 0].copy()
        self.rx_ring.push(rx)
        with self._lock:
            self._latest_rx = rx

    def take_rx(self):
        with self._lock:
            r = self._latest_rx
            self._latest_rx = np.zeros(0, dtype=np.float32)
            return r


# ============================================================================
def build_engines(args):
    if args.modem == "fsk":
        tone0 = float(args.tone0) if args.tone0 else None
        tone1 = float(args.tone1) if args.tone1 else None
        tx = se.FSKEngine(fs=args.fs, bit_rate=args.rate, tone0=tone0, tone1=tone1)
        rx = se.FSKEngine(fs=args.fs, bit_rate=args.rate, tone0=tone0, tone1=tone1)
    elif args.modem == "qpsk":
        tx = se.QPSKEngine(fs=args.fs, bit_rate=args.rate, carrier=args.carrier)
        rx = se.QPSKEngine(fs=args.fs, bit_rate=args.rate, carrier=args.carrier)
    elif args.modem == "cdma":
        tx = se.CDMAEngine(fs=args.fs, bit_rate=100.0, carrier=12800.0)
        rx = se.CDMAEngine(fs=args.fs, bit_rate=100.0, carrier=12800.0)
    elif args.modem == "ofdm":
        cfg = op.wired_config(fs=args.fs) if args.wired else op.acoustic_config(fs=args.fs)
        max_load = op.QAM256 if args.wired else op.QAM64
        tx = se.OFDMEngine(fs=args.fs, cfg=cfg, use_adaptive=args.adaptive, max_loading=max_load)
        rx = se.OFDMEngine(fs=args.fs, cfg=cfg, use_adaptive=args.adaptive, max_loading=max_load)
    elif args.modem == "fhss":
        tx = se.FHSSEngine(fs=args.fs)
        rx = se.FHSSEngine(fs=args.fs)
    else:
        raise ValueError(args.modem)
    return tx, rx


# ============================================================================
class DashboardMPL:
    def __init__(self, args):
        self.args = args
        self.fs = args.fs
        self.tx_engine, self.rx_engine = build_engines(args)
        self.interference = InterferenceSource(fs=self.fs)
        self.audio = AudioEngine(self.tx_engine, self.rx_engine, self.interference,
                                  self.fs, args.blocksize, args.in_device, args.out_device,
                                  args.mode_audio)
        self.fb_recv = None
        self.fb_send = None
        if args.backchannel_listen:
            self.fb_recv = FeedbackReceiver(port=args.backchannel_port)
            self.fb_recv.start()
        if args.backchannel_tx:
            self.fb_send = FeedbackSender(host=args.backchannel_tx, port=args.backchannel_port,
                                           sender_label=args.label, interval_s=0.10)
        # State
        self.last_rx_state: Dict = {}
        self.pe_history: deque = deque(maxlen=600)
        self.cum_history: deque = deque(maxlen=600)
        self._build_fig()

    def _build_fig(self):
        plt.style.use("dark_background")
        if self.args.modem == "ofdm":
            self.fig = plt.figure(figsize=(15, 10))
            gs = self.fig.add_gridspec(4, 4, hspace=0.45, wspace=0.4)
            self.ax_tx = self.fig.add_subplot(gs[0, 0:2])
            self.ax_rx = self.fig.add_subplot(gs[0, 2:4])
            self.ax_spec = self.fig.add_subplot(gs[1, 0:2])
            self.ax_pe = self.fig.add_subplot(gs[1, 2:4])
            self.ax_const: List = []
            for r in range(2):
                for c in range(4):
                    ax = self.fig.add_subplot(gs[2 + r, c])
                    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks([]); ax.set_yticks([])
                    self.ax_const.append(ax)
        else:
            self.fig = plt.figure(figsize=(14, 9))
            gs = self.fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
            self.ax_tx = self.fig.add_subplot(gs[0, 0])
            self.ax_rx = self.fig.add_subplot(gs[0, 1])
            self.ax_spec = self.fig.add_subplot(gs[1, 0])
            self.ax_pe = self.fig.add_subplot(gs[1, 1])
            ax_c = self.fig.add_subplot(gs[2, :])
            ax_c.set_xlim(-1.8, 1.8); ax_c.set_ylim(-1.8, 1.8)
            ax_c.set_aspect('equal'); ax_c.grid(True, alpha=0.3)
            self.ax_const = [ax_c]

        # TX waveform
        self.tx_line, = self.ax_tx.plot([], [], color='#5cb85c', lw=0.8)
        self.ax_tx.set_title("TX waveform")
        self.ax_tx.set_ylim(-1.05, 1.05); self.ax_tx.grid(True, alpha=0.3)

        # RX waveform
        self.rx_line, = self.ax_rx.plot([], [], color='#5bc0de', lw=0.8)
        self.ax_rx.set_title("RX waveform (microphone)")
        self.ax_rx.set_ylim(-1.05, 1.05); self.ax_rx.grid(True, alpha=0.3)

        # Spectrum (log x)
        self.spec_line, = self.ax_spec.plot([], [], color='#f0ad4e', lw=1.0)
        self.ax_spec.set_xscale('log')
        self.ax_spec.set_xlim(20, 20000)
        self.ax_spec.set_ylim(-110, 5)
        self.ax_spec.set_xlabel("Hz")
        self.ax_spec.set_ylabel("dB")
        self.ax_spec.set_title("Spectrum 20-20 kHz")
        self.ax_spec.grid(True, alpha=0.3, which='both')

        # Pe (log y)
        self.pe_line,  = self.ax_pe.plot([], [], color='#d9534f', lw=2, label='Pe (EMA)')
        self.cum_line, = self.ax_pe.plot([], [], color='#5bc0de', lw=2, ls='--', label='Pe (cum)')
        self.ax_pe.set_yscale('log')
        self.ax_pe.set_ylim(1e-5, 1.0)
        self.ax_pe.axhline(0.01, color='red', ls=':', alpha=0.6, label='target 1e-2')
        self.ax_pe.legend(loc='upper right', fontsize=8)
        self.ax_pe.set_title("Running Pe")
        self.ax_pe.grid(True, alpha=0.3, which='both')

        # Constellation scatters
        self.const_scatters = [ax.scatter([], [], s=5, c='white', alpha=0.7) for ax in self.ax_const]
        for i, ax in enumerate(self.ax_const):
            ax.set_title(f"const {i}", fontsize=8)

        # Status text
        self.status_text = self.fig.suptitle(f"{self.tx_engine.name() if self.tx_engine else ''}",
                                              fontsize=12, color='#5cb85c')

    def _on_close(self, ev):
        try:
            self.audio.stop()
        except Exception:
            pass
        if self.fb_recv: self.fb_recv.stop()
        if self.fb_send: self.fb_send.close()

    def _process_rx(self):
        rx_block = self.audio.take_rx()
        if len(rx_block) == 0:
            return
        try:
            state = self.rx_engine.process(rx_block)
            self.last_rx_state = state
        except Exception as e:
            print(f"[rx err] {e}", file=sys.stderr)
            return
        # Backchannel
        if self.fb_send is not None and state.get("snr_per_bin_db") is not None:
            jammer = []
            if self.args.modem == "ofdm" and isinstance(self.rx_engine, se.OFDMEngine):
                rx_recent = self.audio.rx_ring.snapshot(n=int(0.5 * self.fs))
                jammer = detect_jammer_bins(rx_recent, fs=self.fs, n_fft=self.rx_engine.cfg.n,
                                             active_bins=self.rx_engine.cfg.active_bins(), k_sigma=6.0)
            self.fb_send.send(np.asarray(state["snr_per_bin_db"]),
                               jammer_bins=jammer,
                               pe_ema=float(state.get("pe_ema", 0)),
                               pe_cumulative=float(state.get("pe_cumulative", 0)),
                               mode=state.get("mode", "ofdm"))
        if self.fb_recv is not None and self.args.modem == "ofdm" and isinstance(self.tx_engine, se.OFDMEngine):
            pkt = self.fb_recv.latest()
            if pkt is not None:
                snr = np.zeros(self.tx_engine.cfg.n)
                for k, v in enumerate(pkt.snr_per_bin_db):
                    if k < len(snr):
                        snr[k] = v
                self.tx_engine.feed_backchannel(snr_db=snr, jammer_bins=pkt.jammer_bins)

    def _update(self, frame_idx):
        # process rx
        self._process_rx()
        # waveforms
        tx_buf = self.audio.tx_ring.snapshot(n=int(0.10 * self.fs))
        rx_buf = self.audio.rx_ring.snapshot(n=int(0.10 * self.fs))
        x = np.arange(len(tx_buf))
        self.tx_line.set_data(x, tx_buf)
        self.ax_tx.set_xlim(0, max(len(tx_buf), 1))
        x2 = np.arange(len(rx_buf))
        self.rx_line.set_data(x2, rx_buf)
        self.ax_rx.set_xlim(0, max(len(rx_buf), 1))
        # spectrum
        spec_input = self.audio.rx_ring.snapshot(n=8192)
        if len(spec_input) >= 1024:
            n_fft = min(8192, len(spec_input))
            sig = spec_input[-n_fft:] * np.hanning(n_fft)
            P = np.abs(np.fft.rfft(sig)) ** 2
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.fs)
            db = 10 * np.log10(P + 1e-12)
            valid = freqs > 20.0
            self.spec_line.set_data(freqs[valid], db[valid])

        # status / Pe
        s = self.last_rx_state
        pe_ema = float(s.get("pe_ema", 0)) if s else 0.0
        pe_cum = float(s.get("pe_cumulative", 0)) if s else 0.0
        seen = int(s.get("frames_seen", 0)) if s else 0
        locked = int(s.get("frames_locked", 0)) if s else 0
        rate = float(s.get("bit_rate_bps", 0)) if s else 0.0
        snr = float(s.get("snr_db", 0)) if s else 0.0
        self.pe_history.append(max(pe_ema, 1e-6))
        self.cum_history.append(max(pe_cum, 1e-6))
        x3 = np.arange(len(self.pe_history))
        self.pe_line.set_data(x3, list(self.pe_history))
        self.cum_line.set_data(x3, list(self.cum_history))
        self.ax_pe.set_xlim(max(0, len(self.pe_history) - 600), max(len(self.pe_history), 1))

        # constellations
        consts = s.get("constellations", {}) if s else {}
        if self.args.modem == "ofdm":
            data_bins = list(s.get("data_bins", [])) if s else []
            if len(data_bins) >= 16:
                step = len(data_bins) // 16
                show_bins = [data_bins[i * step] for i in range(16)]
            else:
                show_bins = data_bins[:16]
            for k in range(len(self.const_scatters)):
                if k < len(show_bins):
                    pts = consts.get(int(show_bins[k]), np.zeros(0, dtype=np.complex64))
                    if len(pts) > 0:
                        pts_disp = pts[-200:]
                        self.const_scatters[k].set_offsets(np.column_stack([np.real(pts_disp), np.imag(pts_disp)]))
                        self.ax_const[k].set_title(f"bin {show_bins[k]}", fontsize=7)
        else:
            pts = consts.get(0, np.zeros(0, dtype=np.complex64))
            if len(pts) > 0:
                pts_disp = pts[-500:]
                self.const_scatters[0].set_offsets(np.column_stack([np.real(pts_disp), np.imag(pts_disp)]))

        # Title
        color = '#5cb85c' if pe_ema < 0.01 else ('#f0ad4e' if pe_ema < 0.1 else '#d9534f')
        title = (f"{self.tx_engine.name() if self.tx_engine else ''}   "
                 f"Pe(EMA)={pe_ema:.5f}  Pe(cum)={pe_cum:.5f}  "
                 f"frames={locked}/{seen}  rate={rate:.0f} bps  SNR≈{snr:.1f} dB")
        self.status_text.set_text(title)
        self.status_text.set_color(color)
        return [self.tx_line, self.rx_line, self.spec_line, self.pe_line, self.cum_line] + self.const_scatters

    def run(self):
        self.audio.start()
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self._anim = animation.FuncAnimation(self.fig, self._update, interval=80,
                                              blit=False, cache_frame_data=False)
        plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modem", choices=["fsk", "qpsk", "cdma", "ofdm", "fhss"], default="ofdm")
    p.add_argument("--rate", type=float, default=5000.0)
    p.add_argument("--carrier", type=float, default=4800.0)
    p.add_argument("--tone0", type=float, default=None)
    p.add_argument("--tone1", type=float, default=None)
    p.add_argument("--wired", action="store_true")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--fs", type=int, default=DEFAULT_FS)
    p.add_argument("--blocksize", type=int, default=2048)
    p.add_argument("--in-device", type=int, default=None)
    p.add_argument("--out-device", type=int, default=None)
    p.add_argument("--mode-audio", choices=["duplex", "tx_only", "rx_only", "loopback"], default="duplex")
    p.add_argument("--backchannel-listen", action="store_true")
    p.add_argument("--backchannel-tx", default=None)
    p.add_argument("--backchannel-port", type=int, default=9999)
    p.add_argument("--label", default="rx-default")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices()); return 0
    d = DashboardMPL(args)
    d.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
