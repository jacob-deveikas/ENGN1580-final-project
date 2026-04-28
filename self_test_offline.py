from __future__ import annotations

from graded_common import (
    build_fsk_packet, build_qpsk_packet, build_cdma_packet, decode_fsk_capture,
    decode_qpsk_capture, decode_cdma_capture, parse_profiles, prbs_bits
)


def check(label: str, ok: bool, detail: str) -> int:
    print(f"[{label}] {'OK' if ok else 'FAIL'} {detail}")
    return 0 if ok else 1


def main() -> int:
    failures = 0
    for rate, profile in [(50, 'high'), (500, 'high'), (5000, 'wide')]:
        bits = prbs_bits(256)
        wave, _ = build_fsk_packet(bits, rate, parse_profiles(profile))
        r = decode_fsk_capture(wave, rate, 256, parse_profiles(profile))
        failures += check(f"fsk-{rate}-{profile}", bool(r.get('ok')), f"Pe={r.get('pe')} errors={r.get('bit_errors')}")
    for rate in [50, 500, 5000]:
        bits = prbs_bits(256)
        wave, _ = build_qpsk_packet(bits, rate, 7200.0)
        r = decode_qpsk_capture(wave, rate, 256, 7200.0)
        failures += check(f"qpsk-{rate}", bool(r.get('ok')), f"Pe={r.get('pe')} errors={r.get('bit_errors')}")
    bits = prbs_bits(128)
    wave, _ = build_cdma_packet(bits, 100, 12800.0)
    r = decode_cdma_capture(wave, 128, 100, 12800.0)
    failures += check("cdma-100", bool(r.get('ok')), f"Pe={r.get('pe')} errors={r.get('bit_errors')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
