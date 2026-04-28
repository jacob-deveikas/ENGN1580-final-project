#!/usr/bin/env bash
# run_demo.sh - one-stop launcher for the Thursday graded demo.
#
# Usage:
#   ./run_demo.sh self-test              # offline self-tests (run before each demo)
#   ./run_demo.sh dashboard              # full real-time dashboard, OFDM acoustic
#   ./run_demo.sh dashboard-wired        # full dashboard, OFDM wired (90+ kbps)
#   ./run_demo.sh dashboard-adaptive     # OFDM + adaptive + listens for feedback
#   ./run_demo.sh dashboard-rx-feedback IP   # RX side that sends UDP feedback to IP
#   ./run_demo.sh dashboard-fsk          # dashboard with FSK at 5000 bps
#   ./run_demo.sh dashboard-qpsk         # dashboard with QPSK at 5000 bps
#   ./run_demo.sh dashboard-cdma         # dashboard with 64-chip CDMA
#   ./run_demo.sh dashboard-fhss         # dashboard with frequency hopping
#   ./run_demo.sh fsk-sweep              # V5: TX FSK at 50, 500, 5000 bps in sequence
#   ./run_demo.sh qpsk-sweep             # V5: TX QPSK at 50, 500, 5000 bps in sequence
#   ./run_demo.sh cdma                   # V5: TX 64-chip CDMA at 100 bps
#   ./run_demo.sh rx-fsk RATE PROFILE    # V5: receive one FSK packet
#   ./run_demo.sh rx-qpsk RATE [CARRIER] # V5: receive one QPSK packet
#   ./run_demo.sh rx-cdma                # V5: receive one CDMA packet
#   ./run_demo.sh pe-loop-fsk RATE PROFILE COUNT
#   ./run_demo.sh pe-loop-qpsk RATE COUNT
#   ./run_demo.sh interference KIND [FREQ] [GAIN]
#   ./run_demo.sh devices                # list audio devices
#
# Environment overrides:
#   IN_DEVICE=<id>   pin the input device (e.g. for External Microphone)
#   OUT_DEVICE=<id>  pin the output device (e.g. for External Headphones)
#   FS=<rate>        sample rate (default: 44100)

set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
FS=${FS:-44100}
IN_DEV=""
OUT_DEV=""
[[ -n "${IN_DEVICE:-}" ]] && IN_DEV="--in-device ${IN_DEVICE}"
[[ -n "${OUT_DEVICE:-}" ]] && OUT_DEV="--out-device ${OUT_DEVICE}"

V5_DEV=""
[[ -n "${IN_DEVICE:-}" ]] && V5_DEV="--device ${IN_DEVICE}"

cmd="${1:-help}"
shift || true

case "$cmd" in
  help|"")
    sed -n '1,30p' "$0"
    ;;

  self-test)
    echo ">>> Running streaming engine self-test (offline AWGN loopback for all 5 modes)"
    $PY self_test_streaming.py
    echo ""
    echo ">>> Running V5 baseline self-test (file-based)"
    $PY self_test_offline.py
    ;;

  devices)
    $PY -c "import sounddevice as sd; print(sd.query_devices())"
    ;;

  dashboard)
    echo ">>> Launching real-time dashboard, OFDM acoustic mode."
    echo "    Items 1-5 (instrumentation) all visible in this one window."
    echo "    Item 9 (wildcard) = OFDM at ~12 kbps acoustic / can adapt to 30+ kbps."
    $PY dashboard.py --modem ofdm $IN_DEV $OUT_DEV
    ;;

  dashboard-wired)
    echo ">>> Launching real-time dashboard, OFDM WIRED mode."
    echo "    Headline rate: ~80-100 kbps net at Pe < 10^-3 over the cable."
    $PY dashboard.py --modem ofdm --wired --adaptive $IN_DEV $OUT_DEV
    ;;

  dashboard-adaptive)
    echo ">>> Launching real-time dashboard, OFDM + ADAPTIVE + LISTEN for feedback."
    echo "    This is the TX side. Run 'dashboard-rx-feedback <THIS_IP>' on RX laptop."
    $PY dashboard.py --modem ofdm --adaptive --backchannel-listen $IN_DEV $OUT_DEV
    ;;

  dashboard-rx-feedback)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 dashboard-rx-feedback <TX_LAPTOP_IP>"
      exit 2
    fi
    TX_IP="$1"
    echo ">>> Launching real-time dashboard, OFDM RX, sending feedback to $TX_IP"
    $PY dashboard.py --modem ofdm --backchannel-tx "$TX_IP" --label "rx-laptop" $IN_DEV $OUT_DEV
    ;;

  dashboard-fsk)
    RATE=${1:-5000}
    echo ">>> Launching dashboard, FSK at $RATE bps (orthogonal-tone auto-select)"
    $PY dashboard.py --modem fsk --rate "$RATE" $IN_DEV $OUT_DEV
    ;;

  dashboard-qpsk)
    RATE=${1:-5000}
    CARRIER=${2:-4800}
    echo ">>> Launching dashboard, QPSK at $RATE bps, fc=$CARRIER Hz"
    $PY dashboard.py --modem qpsk --rate "$RATE" --carrier "$CARRIER" $IN_DEV $OUT_DEV
    ;;

  dashboard-cdma)
    echo ">>> Launching dashboard, 64-chip CDMA at 100 bps, fc=12.8 kHz"
    $PY dashboard.py --modem cdma $IN_DEV $OUT_DEV
    ;;

  dashboard-fhss)
    echo ">>> Launching dashboard, FHSS over 32 slots @ 50 bps"
    $PY dashboard.py --modem fhss $IN_DEV $OUT_DEV
    ;;

  fsk-sweep)
    PROFILE=${1:-high}
    for r in 50 500 5000; do
      echo ">>> FSK $r bps profile=$PROFILE"
      $PY tx_graded.py --mode fsk --rate "$r" --n-bits 512 --profiles "$PROFILE" $V5_DEV
      echo ">>> sleep 2s before next rate"
      sleep 2
    done
    ;;

  qpsk-sweep)
    CARRIER=${1:-4800}
    for r in 50 500 5000; do
      echo ">>> QPSK $r bps carrier=$CARRIER"
      $PY tx_graded.py --mode qpsk --rate "$r" --n-bits 512 --carrier "$CARRIER" $V5_DEV
      echo ">>> sleep 2s before next rate"
      sleep 2
    done
    ;;

  cdma)
    echo ">>> CDMA 64-chip 100 bps fc=12.8 kHz"
    $PY tx_graded.py --mode cdma --rate 100 --carrier 12800 --n-bits 512 $V5_DEV
    ;;

  rx-fsk)
    RATE=${1:?rate}
    PROFILE=${2:-high}
    $PY rx_graded.py --mode fsk --rate "$RATE" --n-bits 512 --profiles "$PROFILE" $V5_DEV
    ;;

  rx-qpsk)
    RATE=${1:?rate}
    CARRIER=${2:-4800}
    $PY rx_graded.py --mode qpsk --rate "$RATE" --n-bits 512 --carrier "$CARRIER" $V5_DEV
    ;;

  rx-cdma)
    $PY rx_graded.py --mode cdma --rate 100 --carrier 12800 --n-bits 512 $V5_DEV
    ;;

  pe-loop-fsk)
    RATE=${1:?rate}
    PROFILE=${2:-high}
    COUNT=${3:-10}
    $PY pe_rx_loop.py --mode fsk --rate "$RATE" --profiles "$PROFILE" --count "$COUNT" --n-bits 512 $V5_DEV
    ;;

  pe-loop-qpsk)
    RATE=${1:?rate}
    COUNT=${2:-10}
    CARRIER=${3:-4800}
    $PY pe_rx_loop.py --mode qpsk --rate "$RATE" --carrier "$CARRIER" --count "$COUNT" --n-bits 512 $V5_DEV
    ;;

  interference)
    KIND=${1:-sine}
    FREQ=${2:-3000}
    GAIN=${3:-0.20}
    DUR=${4:-30}
    $PY interference_generator.py --kind "$KIND" --freq "$FREQ" --gain "$GAIN" --duration "$DUR" ${OUT_DEVICE:+--device "$OUT_DEVICE"}
    ;;

  *)
    echo "Unknown command: $cmd"
    echo ""
    sed -n '1,30p' "$0"
    exit 2
    ;;
esac
