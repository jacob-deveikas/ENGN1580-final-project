"""UDP backchannel for adaptive modulation.

The receiver sends per-bin SNR estimates and detected jammer bins to the
transmitter every ~100 ms. The transmitter uses a Chow-style threshold lookup
to update the bit-loading map.

This is explicitly allowed by the rubric ("you may use a WLAN back-channel to
relay receiver measurements to the transmitter (a cheat, but allowed for this
part)").

Wire format: a single line of JSON per UDP datagram, no fragmentation. We keep
each datagram under ~1500 bytes by quantizing SNR to 1-decimal floats.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


DEFAULT_PORT = 9999


@dataclass
class FeedbackPacket:
    ts_ms: int
    snr_per_bin_db: List[float]
    jammer_bins: List[int]
    pe_ema: float = 0.0
    pe_cumulative: float = 0.0
    mode: str = "ofdm"
    sender: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "ts_ms": int(self.ts_ms),
            "snr_per_bin_db": [round(float(x), 1) for x in self.snr_per_bin_db],
            "jammer_bins": list(map(int, self.jammer_bins)),
            "pe_ema": float(self.pe_ema),
            "pe_cumulative": float(self.pe_cumulative),
            "mode": str(self.mode),
            "sender": str(self.sender),
        })

    @classmethod
    def from_json(cls, raw: str) -> "FeedbackPacket":
        obj = json.loads(raw)
        return cls(
            ts_ms=int(obj.get("ts_ms", 0)),
            snr_per_bin_db=list(obj.get("snr_per_bin_db", [])),
            jammer_bins=list(obj.get("jammer_bins", [])),
            pe_ema=float(obj.get("pe_ema", 0.0)),
            pe_cumulative=float(obj.get("pe_cumulative", 0.0)),
            mode=str(obj.get("mode", "ofdm")),
            sender=str(obj.get("sender", "")),
        )


class FeedbackSender:
    """RX side: ship feedback to TX every ~100 ms."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 sender_label: str = "rx", interval_s: float = 0.100):
        self.host = host
        self.port = int(port)
        self.sender_label = sender_label
        self.interval_s = float(interval_s)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_ts = 0.0

    def send(self, snr_per_bin_db: np.ndarray, jammer_bins: List[int],
             pe_ema: float = 0.0, pe_cumulative: float = 0.0, mode: str = "ofdm") -> bool:
        now = time.time()
        if now - self._last_ts < self.interval_s:
            return False
        self._last_ts = now
        snr_list = [round(float(x), 1) for x in np.asarray(snr_per_bin_db).tolist()]
        pkt = FeedbackPacket(
            ts_ms=int(now * 1000), snr_per_bin_db=snr_list,
            jammer_bins=list(map(int, jammer_bins)),
            pe_ema=float(pe_ema), pe_cumulative=float(pe_cumulative),
            mode=mode, sender=self.sender_label,
        )
        try:
            self._sock.sendto(pkt.to_json().encode("utf-8"), (self.host, self.port))
            return True
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class FeedbackReceiver:
    """TX side: poll feedback in a background thread."""

    def __init__(self, port: int = DEFAULT_PORT, on_packet: Optional[Callable[[FeedbackPacket], None]] = None):
        self.port = int(port)
        self.on_packet = on_packet
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(("0.0.0.0", self.port))
        except OSError:
            self._sock.bind(("127.0.0.1", self.port))
        self._sock.settimeout(0.5)
        self._latest: Optional[FeedbackPacket] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._packets_received = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                pkt = FeedbackPacket.from_json(data.decode("utf-8"))
            except Exception:
                continue
            with self._lock:
                self._latest = pkt
                self._packets_received += 1
            if self.on_packet is not None:
                try:
                    self.on_packet(pkt)
                except Exception:
                    pass

    def latest(self) -> Optional[FeedbackPacket]:
        with self._lock:
            return self._latest

    @property
    def packets_received(self) -> int:
        with self._lock:
            return self._packets_received


# ----------------------------- jammer detector -------------------------------

def detect_jammer_bins(rx_recent: np.ndarray, fs: int, n_fft: int,
                        active_bins: np.ndarray, k_sigma: float = 6.0) -> List[int]:
    """Return list of FFT-bin indices whose power exceeds the median by k_sigma.

    Uses a single FFT over the most-recent samples. Robust to brief transients.
    """
    if len(rx_recent) < n_fft:
        return []
    seg = rx_recent[-n_fft:] * np.hanning(n_fft)
    P = np.abs(np.fft.fft(seg)) ** 2
    P = P[:n_fft // 2]
    if len(active_bins) == 0:
        return []
    active_in_range = active_bins[active_bins < len(P)]
    if len(active_in_range) == 0:
        return []
    band_power = P[active_in_range]
    med = float(np.median(band_power) + 1e-12)
    mad = float(np.median(np.abs(band_power - med)) + 1e-12)
    sigma = 1.4826 * mad
    threshold = med + k_sigma * sigma
    bad = active_in_range[band_power > threshold].tolist()
    return [int(b) for b in bad]


def smoke_test() -> None:
    """Simple loopback: send 5 packets, receive 5 packets."""
    rx = FeedbackReceiver(port=DEFAULT_PORT)
    rx.start()
    time.sleep(0.05)
    tx = FeedbackSender(host="127.0.0.1", port=DEFAULT_PORT, interval_s=0.0)
    for k in range(5):
        snr = np.full(64, 20.0 + k)
        tx.send(snr, [k, k + 1], pe_ema=0.001 * k, pe_cumulative=0.0005 * k, mode="ofdm")
        time.sleep(0.02)
    time.sleep(0.1)
    print(f"[backchannel] received {rx.packets_received} packets, latest={rx.latest()}")
    tx.close()
    rx.stop()


if __name__ == "__main__":
    smoke_test()
