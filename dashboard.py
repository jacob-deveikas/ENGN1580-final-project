"""Master pyqtgraph dashboard.

One window covers all five Chris-instrumentation rubric items at once:
  1) Real-time TX waveform
  2) Real-time RX waveform
  3) Real-time spectrum 20 Hz - 20 kHz (log-x)
  4) Real-time constellation grid (one per OFDM subcarrier or one big one)
  5) Real-time running-average Pe meter
plus:
  6) Interference generator panel (sine + white + freq slider + amplitude)
  7) Mode selector (FSK / QPSK / CDMA / OFDM / FHSS)
  8) Bit-loading map and adaptive-modulation status
  9) Backchannel status (packets received, latest SNR map)

Audio plumbing:
  - sounddevice.Stream with channels=(1, 2) (mono in, stereo out, mono is
    auto-duplicated to L/R).
  - The TX engine produces float32 mono samples; we duplicate to stereo and
    add the interference generator's contribution.
  - The RX engine consumes mic samples in the audio callback (must be fast)
    and stores them; heavy DSP runs in a QTimer in the GUI thread.

Run examples:
  python dashboard.py --mode ofdm --carrier 4800 --device 1
  python dashboard.py --mode qpsk --rate 5000 --carrier 4800
  python dashboard.py --mode fhss
  python dashboard.py --mode ofdm --backchannel-tx 192.168.1.42
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    HAVE_PG = True
except Exception as _e:
    HAVE_PG = False
    pg = None
    QtCore = None
    QtGui = None
    QtWidgets = None
    print(f"[dashboard] pyqtgraph not installed: {_e}. Falling back to matplotlib live monitor.")
    print("[dashboard] To install: pip install pyqtgraph PyQt6")
    sys.exit(2)

import streaming_engine as se
import ofdm_phy as op
from backchannel import FeedbackReceiver, FeedbackSender, detect_jammer_bins
from prbs15 import BERMeter

DEFAULT_FS = 48000


# ============================================================================
# Interference generator (audio output side)
# ============================================================================
class InterferenceSource:
    def __init__(self, fs: int = DEFAULT_FS):
        self.fs = fs
        self.freq = 3000.0
        self.amp = 0.0
        self.white = False
        self.phase = 0.0
        self.lock = threading.Lock()

    def set_freq(self, hz: float) -> None:
        with self.lock:
            self.freq = max(20.0, min(20000.0, float(hz)))

    def set_amp(self, a: float) -> None:
        with self.lock:
            self.amp = max(0.0, min(1.0, float(a)))

    def set_white(self, on: bool) -> None:
        with self.lock:
            self.white = bool(on)

    def next_block(self, n: int) -> np.ndarray:
        with self.lock:
            f = self.freq
            a = self.amp
            white = self.white
            phase = self.phase
        if a <= 1e-6:
            return np.zeros(n, dtype=np.float32)
        if white:
            sig = (np.random.randn(n) * 0.7).astype(np.float32)
        else:
            inc = 2 * math.pi * f / self.fs
            t = phase + inc * np.arange(n)
            sig = np.sin(t).astype(np.float32)
            with self.lock:
                self.phase = (phase + inc * n) % (2 * math.pi)
        return (a * sig).astype(np.float32)


# ============================================================================
# Audio engine wrapper
# ============================================================================
class AudioEngine:
    """Owns the sounddevice stream and dispatches to TX/RX engines."""

    def __init__(self, tx_engine, rx_engine, interference: InterferenceSource,
                 fs: int = DEFAULT_FS, blocksize: int = 2048,
                 in_device=None, out_device=None,
                 mode: str = "duplex"):
        self.tx_engine = tx_engine
        self.rx_engine = rx_engine
        self.interference = interference
        self.fs = fs
        self.blocksize = int(blocksize)
        self.in_device = in_device
        self.out_device = out_device
        self.mode = mode  # "duplex", "tx_only", "rx_only", "loopback"
        self.tx_ring = se.RingBuffer(int(2 * fs))
        self.rx_ring = se.RingBuffer(int(2 * fs))
        self._stream = None
        self._latest_rx_block = np.zeros(0, dtype=np.float32)
        self._rx_lock = threading.Lock()
        self._loopback_buffer = deque(maxlen=8)
        self.tx_gain = 0.85

    def start(self) -> None:
        import sounddevice as sd
        if self.mode == "tx_only":
            self._stream = sd.OutputStream(
                samplerate=self.fs, blocksize=self.blocksize, channels=2,
                dtype="float32", device=self.out_device,
                callback=self._cb_tx_only, latency="low",
            )
        elif self.mode == "rx_only":
            self._stream = sd.InputStream(
                samplerate=self.fs, blocksize=self.blocksize, channels=1,
                dtype="float32", device=self.in_device,
                callback=self._cb_rx_only, latency="low",
            )
        elif self.mode == "loopback":
            # Internal loopback for offline testing - no audio I/O
            self._stream = None
            self._loopback_thread = threading.Thread(target=self._loopback_loop, daemon=True)
            self._loopback_thread.start()
            return
        else:
            try:
                self._stream = sd.Stream(
                    samplerate=self.fs, blocksize=self.blocksize,
                    channels=(1, 2), dtype="float32",
                    device=(self.in_device, self.out_device),
                    callback=self._cb_duplex, latency="low",
                )
            except Exception as e:
                print(f"[audio] duplex stream open failed: {e}; falling back to mono out")
                self._stream = sd.Stream(
                    samplerate=self.fs, blocksize=self.blocksize,
                    channels=(1, 1), dtype="float32",
                    device=(self.in_device, self.out_device),
                    callback=self._cb_duplex_mono, latency="low",
                )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _build_tx_block(self, n: int) -> np.ndarray:
        tx = np.zeros(n, dtype=np.float32)
        if self.tx_engine is not None:
            try:
                tx = self.tx_engine.next_block(n).astype(np.float32)
            except Exception as e:
                print(f"[audio] tx engine error: {e}")
                tx = np.zeros(n, dtype=np.float32)
        jam = self.interference.next_block(n)
        out = self.tx_gain * tx + jam
        # Soft limit to avoid clipping
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32)

    def _cb_tx_only(self, outdata, frames, time_info, status):
        if status:
            print(f"[audio] tx status: {status}")
        block = self._build_tx_block(frames)
        outdata[:, 0] = block
        if outdata.shape[1] >= 2:
            outdata[:, 1] = block
        self.tx_ring.push(block)

    def _cb_rx_only(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] rx status: {status}")
        rx = indata[:, 0].copy()
        self.rx_ring.push(rx)
        with self._rx_lock:
            self._latest_rx_block = rx

    def _cb_duplex(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"[audio] duplex status: {status}")
        block = self._build_tx_block(frames)
        outdata[:, 0] = block
        if outdata.shape[1] >= 2:
            outdata[:, 1] = block
        self.tx_ring.push(block)
        rx = indata[:, 0].copy()
        self.rx_ring.push(rx)
        with self._rx_lock:
            self._latest_rx_block = rx

    def _cb_duplex_mono(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"[audio] duplex(mono) status: {status}")
        block = self._build_tx_block(frames)
        outdata[:, 0] = block
        self.tx_ring.push(block)
        rx = indata[:, 0].copy()
        self.rx_ring.push(rx)
        with self._rx_lock:
            self._latest_rx_block = rx

    def _loopback_loop(self) -> None:
        """Internal loopback mode for offline testing without audio hardware."""
        while True:
            block = self._build_tx_block(self.blocksize)
            self.tx_ring.push(block)
            self.rx_ring.push(block)
            with self._rx_lock:
                self._latest_rx_block = block.copy()
            # Pace at audio rate
            time.sleep(self.blocksize / self.fs)

    def take_latest_rx(self) -> np.ndarray:
        with self._rx_lock:
            r = self._latest_rx_block
            self._latest_rx_block = np.zeros(0, dtype=np.float32)
            return r


# ============================================================================
# Main dashboard window
# ============================================================================
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Audio SDR Demo - Real-Time Dashboard")
        self.resize(1700, 1000)
        self.args = args
        self.fs = args.fs
        self.blocksize = args.blocksize
        # Engines
        self.tx_engine, self.rx_engine = self._build_engines()
        self.interference = InterferenceSource(fs=self.fs)
        # Audio
        self.audio = AudioEngine(
            tx_engine=self.tx_engine, rx_engine=self.rx_engine,
            interference=self.interference, fs=self.fs, blocksize=self.blocksize,
            in_device=args.in_device, out_device=args.out_device,
            mode=args.mode_audio,
        )
        # Backchannel
        self.fb_recv: Optional[FeedbackReceiver] = None
        self.fb_send: Optional[FeedbackSender] = None
        if args.backchannel_listen:
            self.fb_recv = FeedbackReceiver(port=args.backchannel_port)
            self.fb_recv.start()
        if args.backchannel_tx:
            self.fb_send = FeedbackSender(host=args.backchannel_tx, port=args.backchannel_port,
                                            sender_label=args.label, interval_s=0.10)
        # GUI
        self._build_ui()
        # State
        self.last_rx_state: Dict = {}
        self.tx_buf = np.zeros(int(0.25 * self.fs), dtype=np.float32)
        self.rx_buf = np.zeros(int(0.25 * self.fs), dtype=np.float32)
        self.spec_buf = np.zeros(int(self.fs), dtype=np.float32)
        self.pe_history: deque = deque(maxlen=600)
        self.cumulative_history: deque = deque(maxlen=600)
        self.lock_history: deque = deque(maxlen=600)
        # Start
        self.audio.start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)
        self.process_timer = QtCore.QTimer(self)
        self.process_timer.timeout.connect(self._process_rx)
        self.process_timer.start(60)

    # ------------------------------------------------------------------
    def _build_engines(self):
        a = self.args
        tx_engine, rx_engine = None, None
        if a.modem == "fsk":
            tone0 = float(a.tone0) if a.tone0 else 6600.0
            tone1 = float(a.tone1) if a.tone1 else 7800.0
            band_low, band_high = min(tone0, tone1) - 1000, max(tone0, tone1) + 1000
            tx_engine = se.FSKEngine(fs=a.fs, bit_rate=a.rate, tone0=tone0, tone1=tone1,
                                      band_low=band_low, band_high=band_high)
            rx_engine = se.FSKEngine(fs=a.fs, bit_rate=a.rate, tone0=tone0, tone1=tone1,
                                      band_low=band_low, band_high=band_high)
        elif a.modem == "qpsk":
            tx_engine = se.QPSKEngine(fs=a.fs, bit_rate=a.rate, carrier=a.carrier)
            rx_engine = se.QPSKEngine(fs=a.fs, bit_rate=a.rate, carrier=a.carrier)
        elif a.modem == "cdma":
            tx_engine = se.CDMAEngine(fs=a.fs, bit_rate=100.0, carrier=12800.0)
            rx_engine = se.CDMAEngine(fs=a.fs, bit_rate=100.0, carrier=12800.0)
        elif a.modem == "ofdm":
            cfg = op.wired_config(fs=a.fs) if a.wired else op.acoustic_config(fs=a.fs)
            tx_engine = se.OFDMEngine(fs=a.fs, cfg=cfg, use_adaptive=a.adaptive,
                                       max_loading=op.QAM64 if not a.wired else op.QAM256)
            rx_engine = se.OFDMEngine(fs=a.fs, cfg=cfg, use_adaptive=a.adaptive,
                                       max_loading=op.QAM64 if not a.wired else op.QAM256)
        elif a.modem == "fhss":
            tx_engine = se.FHSSEngine(fs=a.fs)
            rx_engine = se.FHSSEngine(fs=a.fs)
        else:
            raise ValueError(f"unknown modem {a.modem}")
        return tx_engine, rx_engine

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # --- Top status bar -----------------------------------------------
        status_bar = QtWidgets.QHBoxLayout()
        self.status_mode = QtWidgets.QLabel(f"<b>{self.tx_engine.name() if self.tx_engine else 'no-mode'}</b>")
        self.status_mode.setStyleSheet("font-size: 14pt; color: #5cb85c;")
        self.status_pe_ema = QtWidgets.QLabel("Pe(EMA): ---")
        self.status_pe_ema.setStyleSheet("font-size: 13pt; color: #f0ad4e;")
        self.status_pe_cum = QtWidgets.QLabel("Pe(cum): ---")
        self.status_pe_cum.setStyleSheet("font-size: 13pt; color: #5bc0de;")
        self.status_lock = QtWidgets.QLabel("frames: 0/0")
        self.status_lock.setStyleSheet("font-size: 13pt; color: #ddd;")
        self.status_rate = QtWidgets.QLabel("rate: --- bps")
        self.status_rate.setStyleSheet("font-size: 13pt; color: #ddd;")
        self.status_papr = QtWidgets.QLabel("PAPR: --- dB")
        self.status_papr.setStyleSheet("font-size: 13pt; color: #aaa;")
        status_bar.addWidget(self.status_mode)
        status_bar.addStretch(1)
        status_bar.addWidget(self.status_rate)
        status_bar.addWidget(QtWidgets.QLabel(" | "))
        status_bar.addWidget(self.status_pe_ema)
        status_bar.addWidget(QtWidgets.QLabel(" | "))
        status_bar.addWidget(self.status_pe_cum)
        status_bar.addWidget(QtWidgets.QLabel(" | "))
        status_bar.addWidget(self.status_lock)
        status_bar.addWidget(QtWidgets.QLabel(" | "))
        status_bar.addWidget(self.status_papr)
        root.addLayout(status_bar)

        # --- Main pyqtgraph layout ----------------------------------------
        gv = pg.GraphicsLayoutWidget()
        root.addWidget(gv, 8)

        # Row 1: TX waveform | RX waveform
        self.tx_plot = gv.addPlot(title="TX Waveform")
        self.tx_plot.setYRange(-1.05, 1.05)
        self.tx_plot.showGrid(x=True, y=True, alpha=0.3)
        self.tx_plot.setLabel('bottom', 'Time (samples)')
        self.tx_curve = self.tx_plot.plot(pen=pg.mkPen('#5cb85c', width=1.0))

        self.rx_plot = gv.addPlot(title="RX Waveform (Microphone)")
        self.rx_plot.setYRange(-1.05, 1.05)
        self.rx_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rx_plot.setLabel('bottom', 'Time (samples)')
        self.rx_curve = self.rx_plot.plot(pen=pg.mkPen('#5bc0de', width=1.0))
        gv.nextRow()

        # Row 2: Spectrum | Pe meter
        self.spec_plot = gv.addPlot(title="Spectrum 20 Hz - 20 kHz (log-x)")
        self.spec_plot.setLogMode(x=True, y=False)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spec_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spec_plot.setLabel('left', 'Power (dB)')
        self.spec_plot.setYRange(-110, 5)
        self.spec_plot.setXRange(np.log10(20), np.log10(20000))
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen('#f0ad4e', width=1.0))
        # Active band region
        self.band_region = pg.LinearRegionItem([np.log10(1000), np.log10(9500)],
                                                  brush=pg.mkBrush(80, 200, 80, 30),
                                                  movable=False)
        self.spec_plot.addItem(self.band_region)

        self.pe_plot = gv.addPlot(title="Running-Average Pe (EMA + cumulative + lock-rate)")
        self.pe_plot.setLogMode(x=False, y=True)
        self.pe_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pe_plot.setLabel('left', 'Pe (log)')
        self.pe_plot.setYRange(-5, 0)   # 1e-5 to 1.0
        self.pe_curve = self.pe_plot.plot(pen=pg.mkPen('#d9534f', width=2), name="Pe (EMA)")
        self.pe_cum_curve = self.pe_plot.plot(pen=pg.mkPen('#5bc0de', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                                name="Pe (cumulative)")
        # Reference line at Pe = 0.01
        ref = pg.InfiniteLine(pos=np.log10(0.01), angle=0,
                                pen=pg.mkPen(color=(255, 80, 80), style=QtCore.Qt.PenStyle.DashLine))
        self.pe_plot.addItem(ref)
        gv.nextRow()

        # Row 3: Constellation grid (4x4 for OFDM, big-single for others)
        self.const_grid = gv.addLayout(rowspan=2, colspan=2)
        self.const_plots: List = []
        self.const_scatters: List = []
        self.const_labels: List = []
        n_panels = 16 if self.args.modem == "ofdm" else 1
        cols = 4 if n_panels == 16 else 1
        for k in range(n_panels):
            p = self.const_grid.addPlot()
            p.setRange(xRange=(-1.6, 1.6), yRange=(-1.6, 1.6))
            p.setAspectLocked(True)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.hideButtons()
            scat = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(255, 255, 255, 180), size=3)
            p.addItem(scat)
            label = pg.TextItem("", color=(200, 220, 255), anchor=(0, 0))
            label.setPos(-1.5, 1.4)
            p.addItem(label)
            self.const_plots.append(p)
            self.const_scatters.append(scat)
            self.const_labels.append(label)
            if (k + 1) % cols == 0:
                self.const_grid.nextRow()

        # --- Controls panel -----------------------------------------------
        controls = QtWidgets.QGroupBox("Interference Generator + Modem Controls")
        controls.setStyleSheet("QGroupBox { font-size: 12pt; }")
        cl = QtWidgets.QGridLayout(controls)
        # Row 0: interference freq slider
        cl.addWidget(QtWidgets.QLabel("Sine freq (Hz):"), 0, 0)
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 1000)   # log mapped
        self.freq_slider.setValue(self._freq_to_slider(3000.0))
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        cl.addWidget(self.freq_slider, 0, 1, 1, 3)
        self.freq_label = QtWidgets.QLabel("3000 Hz")
        self.freq_label.setMinimumWidth(80)
        cl.addWidget(self.freq_label, 0, 4)

        cl.addWidget(QtWidgets.QLabel("Amplitude:"), 1, 0)
        self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(0)
        self.amp_slider.valueChanged.connect(self._on_amp_changed)
        cl.addWidget(self.amp_slider, 1, 1, 1, 3)
        self.amp_label = QtWidgets.QLabel("0.00")
        cl.addWidget(self.amp_label, 1, 4)

        self.white_check = QtWidgets.QCheckBox("White noise (overrides sine)")
        self.white_check.stateChanged.connect(self._on_white_changed)
        cl.addWidget(self.white_check, 2, 0, 1, 2)

        self.jammer_off_btn = QtWidgets.QPushButton("Jammer OFF")
        self.jammer_off_btn.clicked.connect(self._jammer_off)
        cl.addWidget(self.jammer_off_btn, 2, 2)
        self.jammer_max_btn = QtWidgets.QPushButton("Jammer @ 0.30 amp")
        self.jammer_max_btn.clicked.connect(self._jammer_max)
        cl.addWidget(self.jammer_max_btn, 2, 3)

        # Row 3: status of bit-loading map / backchannel
        self.bitload_label = QtWidgets.QLabel("loading: ---")
        self.bitload_label.setWordWrap(True)
        cl.addWidget(self.bitload_label, 3, 0, 1, 5)

        # Row 4: backchannel status
        self.bc_label = QtWidgets.QLabel("backchannel: idle")
        cl.addWidget(self.bc_label, 4, 0, 1, 5)

        root.addWidget(controls, 2)

    # ------------------------------------------------------------------
    def _freq_to_slider(self, hz: float) -> int:
        # log map from 20 Hz (slider 0) to 20000 Hz (slider 1000)
        if hz <= 20:
            return 0
        if hz >= 20000:
            return 1000
        return int(round(1000 * math.log10(hz / 20.0) / math.log10(1000.0)))

    def _slider_to_freq(self, v: int) -> float:
        return 20.0 * (10.0 ** (v * math.log10(1000.0) / 1000.0))

    def _on_freq_changed(self, v: int):
        f = self._slider_to_freq(v)
        self.interference.set_freq(f)
        self.freq_label.setText(f"{f:.0f} Hz")

    def _on_amp_changed(self, v: int):
        a = v / 100.0
        self.interference.set_amp(a)
        self.amp_label.setText(f"{a:.2f}")

    def _on_white_changed(self, state):
        self.interference.set_white(state == QtCore.Qt.CheckState.Checked.value)

    def _jammer_off(self):
        self.amp_slider.setValue(0)

    def _jammer_max(self):
        self.amp_slider.setValue(30)

    # ------------------------------------------------------------------
    def _process_rx(self):
        """Pull RX samples from the audio ring, run them through the rx engine."""
        rx_block = self.audio.take_latest_rx()
        if len(rx_block) == 0:
            # Pull from rx_ring snapshot if no recent block
            return
        try:
            state = self.rx_engine.process(rx_block)
        except Exception as e:
            print(f"[dashboard] rx process error: {e}")
            return
        self.last_rx_state = state
        # Backchannel: send feedback if available
        if self.fb_send is not None and state.get("snr_per_bin_db") is not None:
            jammer = self._detect_jammer_from_spectrum()
            self.fb_send.send(
                snr_per_bin_db=np.asarray(state["snr_per_bin_db"]),
                jammer_bins=jammer,
                pe_ema=float(state.get("pe_ema", 1.0)),
                pe_cumulative=float(state.get("pe_cumulative", 1.0)),
                mode=str(state.get("mode", "ofdm")),
            )
        # Backchannel: receive feedback if available
        if self.fb_recv is not None:
            pkt = self.fb_recv.latest()
            if pkt is not None and self.args.modem == "ofdm" and isinstance(self.tx_engine, se.OFDMEngine):
                snr = np.zeros(self.tx_engine.cfg.n)
                for k, v in enumerate(pkt.snr_per_bin_db):
                    if k < len(snr):
                        snr[k] = v
                self.tx_engine.feed_backchannel(snr_db=snr, jammer_bins=pkt.jammer_bins)

    def _detect_jammer_from_spectrum(self) -> List[int]:
        if self.args.modem != "ofdm" or not isinstance(self.rx_engine, se.OFDMEngine):
            return []
        rx_recent = self.audio.rx_ring.snapshot(n=int(0.5 * self.fs))
        active = self.rx_engine.cfg.active_bins()
        return detect_jammer_bins(rx_recent, fs=self.fs, n_fft=self.rx_engine.cfg.n,
                                    active_bins=active, k_sigma=6.0)

    # ------------------------------------------------------------------
    def _tick(self):
        # Update waveforms
        self.tx_buf = self.audio.tx_ring.snapshot(n=int(0.10 * self.fs))
        self.rx_buf = self.audio.rx_ring.snapshot(n=int(0.10 * self.fs))
        self.tx_curve.setData(self.tx_buf)
        self.rx_curve.setData(self.rx_buf)

        # Spectrum
        spec_input = self.audio.rx_ring.snapshot(n=8192)
        if len(spec_input) >= 1024:
            n_fft = min(8192, len(spec_input))
            sig = spec_input[-n_fft:] * np.hanning(n_fft)
            P = np.abs(np.fft.rfft(sig)) ** 2
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.fs)
            db = 10 * np.log10(P + 1e-12)
            valid = freqs > 20.0
            f_disp = freqs[valid]
            d_disp = db[valid]
            self.spec_curve.setData(np.log10(f_disp), d_disp)

        # Pe / state
        s = self.last_rx_state
        pe_ema = float(s.get("pe_ema", 0.0)) if s else 0.0
        pe_cum = float(s.get("pe_cumulative", 0.0)) if s else 0.0
        seen = int(s.get("frames_seen", 0)) if s else 0
        locked = int(s.get("frames_locked", 0)) if s else 0
        rate = float(s.get("bit_rate_bps", 0.0)) if s else 0.0
        snr = float(s.get("snr_db", 0.0)) if s else 0.0
        papr = float(s.get("papr_db", 0.0)) if s else 0.0
        self.status_pe_ema.setText(f"Pe(EMA): {pe_ema:.5f}")
        # Color the EMA based on whether we're meeting Pe<0.01
        if pe_ema < 0.01:
            self.status_pe_ema.setStyleSheet("font-size: 13pt; color: #5cb85c;")
        elif pe_ema < 0.1:
            self.status_pe_ema.setStyleSheet("font-size: 13pt; color: #f0ad4e;")
        else:
            self.status_pe_ema.setStyleSheet("font-size: 13pt; color: #d9534f;")
        self.status_pe_cum.setText(f"Pe(cum): {pe_cum:.5f}  [{(s.get('bermeter').total_errors if s.get('bermeter') else 0)} / {(s.get('bermeter').total_bits if s.get('bermeter') else 0)} bits]")
        self.status_lock.setText(f"frames: {locked}/{seen}  SNR≈{snr:.1f} dB")
        self.status_rate.setText(f"rate: {rate:.0f} bps")
        if papr > 0:
            self.status_papr.setText(f"PAPR: {papr:.1f} dB")
        else:
            self.status_papr.setText("PAPR: ---")

        # Pe history log plot
        self.pe_history.append(max(pe_ema, 1e-6))
        self.cumulative_history.append(max(pe_cum, 1e-6))
        x = np.arange(len(self.pe_history))
        # log10
        self.pe_curve.setData(x, np.log10(np.array(self.pe_history)))
        self.pe_cum_curve.setData(x, np.log10(np.array(self.cumulative_history)))

        # Constellations
        consts = s.get("constellations", {}) if s else {}
        if self.args.modem == "ofdm":
            data_bins = s.get("data_bins", []) if s else []
            data_bins = list(data_bins)
            # Pick 16 evenly-spaced data bins for display
            if len(data_bins) >= 16:
                step = len(data_bins) // 16
                show_bins = [data_bins[i * step] for i in range(16)]
            else:
                show_bins = data_bins[:16]
            for k, b in enumerate(show_bins[:16]):
                pts = consts.get(int(b), np.zeros(0, dtype=np.complex64))
                if len(pts) > 0:
                    pts_disp = pts[-200:]
                    self.const_scatters[k].setData(np.real(pts_disp), np.imag(pts_disp))
                    bps = int(s.get("loading", np.zeros(self.rx_engine.cfg.n))[b]) if s else 0
                    label = f"bin {b} ({op.estimate_throughput(self.rx_engine.cfg, op.BitLoadingMap.uniform(self.rx_engine.cfg, bps))['delta_f_hz']*b:.0f} Hz, {bps} b/s)"
                    self.const_labels[k].setText(f"bin{b} {bps}b")
            # Bit-loading status
            loading = s.get("loading", []) if s else []
            if len(loading) > 0:
                active = s.get("active_bins", [])
                if len(active):
                    bps_total = int(np.sum(loading))
                    n_off = int(np.sum(np.array(loading)[active] == 0))
                    n_active = len(active)
                    self.bitload_label.setText(
                        f"<b>bit-loading:</b> {bps_total} bits/symbol  ({n_active - n_off}/{n_active} active)  "
                        f"jamming → {n_off} bins masked off"
                    )
        else:
            # Single big constellation
            pts = consts.get(0, np.zeros(0, dtype=np.complex64))
            if len(pts) > 0:
                pts_disp = pts[-500:]
                self.const_scatters[0].setData(np.real(pts_disp), np.imag(pts_disp))

        # Backchannel status
        bc = []
        if self.fb_recv is not None:
            pkt = self.fb_recv.latest()
            if pkt is not None:
                bc.append(f"RX→TX: {self.fb_recv.packets_received} pkts; latest pe={pkt.pe_ema:.4f} jammer={len(pkt.jammer_bins)} bins")
            else:
                bc.append("listening for feedback (no packets yet)")
        if self.fb_send is not None:
            bc.append(f"TX→{self.args.backchannel_tx}:{self.args.backchannel_port}")
        if bc:
            self.bc_label.setText("backchannel: " + " | ".join(bc))

    def closeEvent(self, ev):
        try:
            self.audio.stop()
        except Exception:
            pass
        if self.fb_recv is not None:
            self.fb_recv.stop()
        if self.fb_send is not None:
            self.fb_send.close()
        super().closeEvent(ev)


# ============================================================================
# Entry point
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Audio SDR Demo - real-time pyqtgraph dashboard")
    p.add_argument("--modem", choices=["fsk", "qpsk", "cdma", "ofdm", "fhss"], default="ofdm")
    p.add_argument("--rate", type=float, default=5000.0, help="bit rate for FSK / QPSK")
    p.add_argument("--carrier", type=float, default=4800.0, help="carrier for QPSK")
    p.add_argument("--tone0", type=float, default=None, help="FSK low tone")
    p.add_argument("--tone1", type=float, default=None, help="FSK high tone")
    p.add_argument("--wired", action="store_true", help="use wide-band wired OFDM config")
    p.add_argument("--adaptive", action="store_true", help="OFDM adaptive bit-loading")
    p.add_argument("--fs", type=int, default=DEFAULT_FS)
    p.add_argument("--blocksize", type=int, default=2048)
    p.add_argument("--in-device", type=int, default=None)
    p.add_argument("--out-device", type=int, default=None)
    p.add_argument("--mode-audio", choices=["duplex", "tx_only", "rx_only", "loopback"], default="duplex")
    p.add_argument("--backchannel-listen", action="store_true",
                    help="(TX side) listen for UDP feedback packets and apply to bit-loading")
    p.add_argument("--backchannel-tx", default=None,
                    help="(RX side) IP address of the TX laptop to send feedback to")
    p.add_argument("--backchannel-port", type=int, default=9999)
    p.add_argument("--label", default="rx-default")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOption("background", "#101418")
    pg.setConfigOption("foreground", "#dddddd")
    win = Dashboard(args)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
