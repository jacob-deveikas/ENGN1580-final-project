================================================================
   AUDIO SDR DEMO — START HERE — READ THIS FIRST
================================================================

You are looking at the final package for Chris's audio-channel SDR
demo on Thursday April 30, 2026. The package contains:

  - V5 baseline (FSK / QPSK / 64-chip CDMA) - fully tested
  - Custom OFDM modem (the "wildcard" highest-rate slot)
  - Real-time pyqtgraph dashboard (covers all 5 instrumentation items)
  - UDP adaptive-modulation backchannel (Chris-allowed cheat)
  - Frequency-hopping bonus

================================================================
WHAT TO READ AND WHEN
================================================================

   TODAY (Monday April 27)        ->  this file (_README_FIRST.txt)
                                       INSTALL_AND_PREP.md (next)

   TOMORROW (Tuesday April 28)    ->  LAB_DAY_TUESDAY.txt
                                       <-- THE LAB-DAY PLAN. SPOON-FED.
                                       <-- READ IT TONIGHT. BRING IT TO LAB.

   WEDNESDAY (April 29)           ->  MEGA_CHEAT_SHEET.txt
                                       (rehearse Thursday's demo)
                                       NO NEW CODE WEDNESDAY!

   THURSDAY (April 30) demo day   ->  MEGA_CHEAT_SHEET.txt
                                       (the one-page-per-rubric-item)

================================================================
TODAY: INSTALL AND PREP — DO BEFORE BED
================================================================

ON BOTH LAPTOPS (yours and your friend's):

1. Copy this entire folder to the laptop (USB stick, AirDrop,
   GitHub, whatever works).

2. cd into the folder.

3. pip install -r requirements.txt

4. Run the offline self-tests:
      ./run_demo.sh self-test
   Must say "ALL OK" twice (once for streaming engines, once for V5
   baseline). If it does not, paste the failure into Claude Code and
   we fix it tonight.

5. Run the cross-laptop hash check:
      python3 bitstream_check.py --n-bits 512 --seed 1580
   Both laptops must print the same bit_hash and first64_bits.
   If they don't match, your Python/NumPy versions diverged and
   you'll get random Pe results tomorrow. Fix it tonight.

6. Identify your audio device IDs:
      ./run_demo.sh devices
   Note down the index of "External Headphones" and "External
   Microphone". Set them as environment variables in your shell:
      export OUT_DEVICE=<your-out-id>
      export IN_DEVICE=<your-in-id>

7. macOS lockdown checklist (top of MEGA_CHEAT_SHEET.txt):
   - System Settings -> Bluetooth -> OFF
   - System Settings -> Focus -> Do Not Disturb ON
   - Control Center -> Mic Mode -> Standard (NOT Voice Isolation)
   - Quit Zoom / Teams / FaceTime / Discord / Music
   - sudo mdutil -a -i off  (Spotlight off)
   - caffeinate -dimsu &  (no sleep)
   - Plug in power adapter
   - Sound output volume 60% (NOT max)
   - Sound input volume MAX

8. PACK YOUR BAG:
   - laptop + charger
   - 3.5mm cable
   - Cubilux 3.5mm attenuator (-30 dB)
   - MillSO TRRS splitter (CTIA pinout)
   - USB audio interface as backup
   - this printed cheat sheet

9. Tell your friend to do all 8 of these on his laptop tonight too.

================================================================
TOMORROW (Tuesday April 28): READ LAB_DAY_TUESDAY.txt
================================================================

Tomorrow's plan is in LAB_DAY_TUESDAY.txt. It has three phases:

   Phase 1: single-laptop cable loopback     (~30 min, your laptop)
   Phase 2: two-laptop wired test            (~60 min)
   Phase 3: acoustic test                    (~30 min)
   Phase 4: iterate fixes with Claude

Every command for both laptops is spelled out. Your friend just
copies the >> TX commands. You run the >> RX commands.

================================================================
ONE-COMMAND OPERATIONS (the launcher)
================================================================

EVERYTHING goes through ./run_demo.sh. Run it with no arguments
to see the menu, or:

  ./run_demo.sh self-test            offline self-test
  ./run_demo.sh devices              list audio devices
  ./run_demo.sh dashboard            full dashboard, OFDM acoustic
  ./run_demo.sh dashboard-wired      full dashboard, OFDM wired
  ./run_demo.sh dashboard-adaptive   OFDM + adaptive bit-loading
  ./run_demo.sh dashboard-fsk RATE   dashboard with FSK
  ./run_demo.sh dashboard-qpsk RATE  dashboard with QPSK
  ./run_demo.sh dashboard-cdma       dashboard with CDMA
  ./run_demo.sh dashboard-fhss       dashboard with frequency hopping
  ./run_demo.sh fsk-sweep PROFILE    V5 FSK 50/500/5000 in sequence
  ./run_demo.sh qpsk-sweep CARRIER   V5 QPSK 50/500/5000 in sequence
  ./run_demo.sh cdma                 V5 CDMA 100 bps
  ./run_demo.sh rx-fsk RATE PROF     receive one FSK packet
  ./run_demo.sh rx-qpsk RATE [FC]    receive one QPSK packet
  ./run_demo.sh rx-cdma              receive one CDMA packet
  ./run_demo.sh pe-loop-fsk RATE PROF COUNT
  ./run_demo.sh pe-loop-qpsk RATE COUNT
  ./run_demo.sh interference KIND FREQ GAIN

================================================================
WHAT EACH RUBRIC ITEM USES (one-line per item)
================================================================

Item 1-5 (instrumentation):  ./run_demo.sh dashboard
Item 6   (QPSK 50/500/5000): ./run_demo.sh qpsk-sweep
Item 7   (FSK  50/500/5000): ./run_demo.sh fsk-sweep
Item 8   (CDMA 100 bps):     ./run_demo.sh cdma
Item 9   (wildcard, OFDM):   ./run_demo.sh dashboard / dashboard-wired
Item 10  (adaptive):         ./run_demo.sh dashboard-adaptive
Bonus    (FHSS):             ./run_demo.sh dashboard-fhss

================================================================
ARTIFACTS — WHAT EVERY RUN PRODUCES
================================================================

Every command auto-creates a timestamped folder:
   runs/20260428_120000_rx_fsk_500bps_001/

Inside you get:
   manifest.json                 - exact command run
   tx_*.wav                      - transmitted waveform (TX side)
   tx_*_meta.json                - TX settings
   tx_waveform.png, tx_spectrum.png - TX previews (with --preview)
   rx_*_capture.wav              - microphone recording
   rx_*_result.json              - decode result, Pe, errors, sync
   qpsk_constellation.png        - constellation plot (QPSK)
   pe_loop_summary.json          - running Pe over many packets
   channel_response.{wav,png,csv,json} - sweep measurement
   ambient_choice.json           - ambient scan recommendation

When something breaks, paste me the rx_*_result.json. I read it
and tell you what to do next.

================================================================
RULES THAT CANNOT BE VIOLATED
================================================================

These are Chris's explicit rules. Do not break them.

   NO source compression.
   NO channel coding (Reed-Solomon, convolutional, LDPC, GRAND, etc.)
   NO repetition coding for Pe reduction.

What IS allowed:
   - Preamble (synchronization, not coding)
   - Sync sequence (training, not coding)
   - Pilot/training symbols (channel estimation, not coding)
   - Channel measurement and adaptation (carrier/profile choice)
   - PRBS test bitstream (test source, not coding)
   - WLAN backchannel for adaptive modulation (Chris explicitly allowed)

The PRBS-15 LFSR is a TEST stream. The receiver knows the seed
and regenerates the expected bits to compute Pe. This is NOT FEC.

================================================================
FILE INVENTORY
================================================================

DOCS:
   _README_FIRST.txt              <-- you are here
   LAB_DAY_TUESDAY.txt            tomorrow's spoon-fed plan
   MEGA_CHEAT_SHEET.txt           Thursday's demo cheat sheet
   WIRING_GUIDE_CUBILUX_MILLSO.txt how to wire the laptops
   FAILURE_DEBUG_PROTOCOL.txt     quick debug ref

LAUNCHER:
   run_demo.sh                    one-stop launcher
   requirements.txt               pip dependencies

CORE PHYSICAL-LAYER MODULES (new):
   ofdm_phy.py                    OFDM modulator + demodulator
   streaming_engine.py            continuous TX/RX engines
   prbs15.py                      PRBS-15 LFSR + BERMeter
   backchannel.py                 UDP feedback for adaptive
   fhss_phy.py                    frequency hopping
   dashboard.py                   pyqtgraph real-time dashboard

V5 BASELINE (unchanged, all V5 commands still work):
   graded_common.py               shared FSK/QPSK/CDMA code
   tx_graded.py                   transmit one packet
   rx_graded.py                   receive one packet, compute Pe
   tx_loop_graded.py              repeated TX for running-Pe demo
   pe_rx_loop.py                  repeated RX for running-Pe demo
   auto_test_graded.py            single-laptop TX+RX in subprocess
   measure_channel.py             channel sweep + recommendation
   interference_generator.py      standalone jammer
   live_monitor.py                standalone matplotlib live RX
   qpsk_constellation_live.py     standalone matplotlib QPSK constel.

OFFLINE TESTS (run these before each demo):
   self_test_streaming.py         AWGN loopback for new modes
   self_test_offline.py           V5 file-based loopback
   bitstream_check.py             cross-laptop PRBS hash check

DEBUG HELPERS:
   install_check.py               import sanity
   device_doctor.py               PortAudio/CoreAudio sanity
   diagnose_capture.py            offline WAV decode debugger

UTILITY:
   run_utils.py                   timestamped run folder helpers

================================================================
KEY NUMBERS TO MEMORIZE (Chris will ask)
================================================================

PRBS-15 polynomial:       x^15 + x^14 + 1  (ITU-T O.151)
PRBS-15 period:           32767 bits
CDMA processing gain:     10*log10(64) = 18.06 dB
CDMA chip rate:           100 bps * 64 chips = 6400 chips/s
CDMA carrier:             12.8 kHz
OFDM N (FFT size):        1024
OFDM CP (cyclic prefix):  128 samples = 2.9 ms
OFDM Δf:                  fs/N = 44100/1024 = 43.07 Hz
OFDM symbol period:       (N+CP)/fs = 26.1 ms
OFDM acoustic band:       1.0 - 9.5 kHz, 197 active subcarriers
OFDM wired band:          0.35 - 19.5 kHz, ~450 active
Backchannel cadence:      every 100 ms, JSON over UDP, port 9999
SNR thresholds (Chow):    BPSK 6.5 dB, QPSK 9 dB, 16-QAM 14.5 dB,
                          64-QAM 20 dB, 256-QAM 26 dB

Headline rate claims:
   acoustic OFDM QPSK :  ~12 kbps
   acoustic OFDM 16QAM:  ~25 kbps  (with adaptation in good SNR)
   wired OFDM 16QAM   :  ~55 kbps
   wired OFDM 64QAM   :  ~84 kbps  (the headline)
   acoustic Pe target :  < 1e-3 (rubric line is 1e-2)

================================================================
WHY THIS PACKAGE AND NOT GNU RADIO / QUIET / GGWAVE
================================================================

GNU Radio:    macOS install fragile, BER demo broken in current
              versions, building flowgraphs in 3 days = no.
ggwave:       Reed-Solomon FEC intertwined with init - banned by
              Chris.
libquiet:     C-level cffi binding, harder to explain line-by-line
              when Chris asks "show me the code".

This package is pure Python, easy to read, every choice defensible.

================================================================
WHAT NOT TO DO
================================================================

- No new code after Wednesday noon.
- Do not enable Voice Isolation in Mic Mode. Standard only.
- Do not run system volume at max - it triggers the limiter and
  blows up QAM constellations.
- Do not skip the bitstream_check.py hash compare.
- Do not promise Chris a number you have not measured.
- Do not panic if one mode misbehaves at the lab.  Your friend's
  laptop's audio path or volume is the most likely cause.
- Do not connect to eduroam or the campus WiFi for the backchannel
  test - they isolate clients. Use a phone hotspot or USB-C ethernet.

================================================================
SUPPORT
================================================================

When something breaks tomorrow at the lab, paste me:
   1. The exact command, both sides.
   2. Last 30 lines of your RX terminal.
   3. cat runs/<latest>/rx_*_result.json
   4. Optional: python3 diagnose_capture.py runs/<latest>/rx_*_capture.wav

I'll read it and either tell you to twist a knob or push a fix.

================================================================
                  GOOD LUCK.  YOU GOT THIS.
================================================================
