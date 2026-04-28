from __future__ import annotations
import argparse, json
import numpy as np
from graded_common import (
    load_wav, decode_fsk_capture, decode_qpsk_capture, decode_cdma_capture, parse_profiles,
    find_preamble_candidates, prbs_bits, bitstream_hash, FS, CDMA_CARRIER, CDMA_RATE
)

p = argparse.ArgumentParser(description='Offline diagnostic for a saved receiver capture. Use when Pe is bad.')
p.add_argument('--wav-in', required=True)
p.add_argument('--mode', choices=['fsk','qpsk','cdma'], required=True)
p.add_argument('--rate', type=float, default=None)
p.add_argument('--n-bits', type=int, default=512)
p.add_argument('--seed', type=int, default=1580)
p.add_argument('--profiles', default='high')
p.add_argument('--carrier', type=float, default=None)
p.add_argument('--json-out', default=None)
args = p.parse_args()

x = load_wav(args.wav_in)
rms = float(np.sqrt(np.mean(x*x)+1e-12))
peak = float(np.max(np.abs(x))) if len(x) else 0.0
clip_frac = float(np.mean(np.abs(x) > 0.98)) if len(x) else 0.0
bits = prbs_bits(args.n_bits, args.seed)
print(f'[diag] wav={args.wav_in} samples={len(x)} seconds={len(x)/FS:.3f} rms={rms:.6g} peak={peak:.6g} clip_frac={clip_frac:.6f}')
print(f'[diag] expected seed={args.seed} n_bits={args.n_bits} bit_hash={bitstream_hash(bits)} first64={bits[:64]}')
print('[diag] preamble candidates:', find_preamble_candidates(x, threshold=0.0, max_candidates=5))
rate = args.rate if args.rate is not None else (CDMA_RATE if args.mode == 'cdma' else 500.0)
if args.mode == 'fsk':
    result = decode_fsk_capture(x, rate, args.n_bits, parse_profiles(args.profiles), seed=args.seed)
elif args.mode == 'qpsk':
    carrier = args.carrier if args.carrier is not None else 7200.0
    result = decode_qpsk_capture(x, rate, args.n_bits, carrier, seed=args.seed, search=False)
else:
    carrier = args.carrier if args.carrier is not None else CDMA_CARRIER
    result = decode_cdma_capture(x, args.n_bits, rate=rate, carrier=carrier, seed=args.seed)
print(f"[diag] result ok={result.get('ok')} reason={result.get('reason')} pe={result.get('pe')} errors={result.get('bit_errors')}/{result.get('n_bits')} sync={result.get('sync_score', result.get('score'))}")
if 'profile' in result: print(f"[diag] profile={result['profile']}")
if 'carrier' in result: print(f"[diag] carrier={result['carrier']}")
if args.json_out:
    clean = {k:v for k,v in result.items() if k not in {'received_bits','expected_bits','constellation','sync_constellation'}}
    with open(args.json_out, 'w', encoding='utf-8') as f: json.dump(clean, f, indent=2)
    print(f'[diag] wrote {args.json_out}')
