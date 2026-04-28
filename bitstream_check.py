from __future__ import annotations
import argparse
from graded_common import prbs_bits, bitstream_hash

p=argparse.ArgumentParser(description='Print the exact deterministic uncoded PRBS test stream identity. Run on BOTH laptops before graded demos.')
p.add_argument('--n-bits', type=int, default=512)
p.add_argument('--seed', type=int, default=1580)
args=p.parse_args()
bits=prbs_bits(args.n_bits,args.seed)
print(f'n_bits={args.n_bits}')
print(f'seed={args.seed}')
print(f'bit_hash={bitstream_hash(bits)}')
print(f'first64={bits[:64]}')
print('Both laptops must print the exact same bit_hash and first64 for Pe tests to be meaningful.')
