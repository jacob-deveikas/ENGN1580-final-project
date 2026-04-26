RADIO MOG SUITE

This package is split into two lanes because Chris changed the rules.

LANE A: THURSDAY GRADED, NO CODING, NO COMPRESSION
Use these files:
  tx_graded.py
  rx_graded.py
  auto_test_graded.py
  pe_rx_loop.py
  tx_loop_graded.py
  measure_channel.py
  live_monitor.py
  interference_generator.py

These are uncoded physical-layer demonstrations. They transmit a deterministic PRBS bit stream so the receiver can compute P_e directly. There is no source compression and no error-correcting channel code.

Supported graded modes:
  FSK at 50, 500, 5000 bits/s
  QPSK/4-QAM at 50, 500, 5000 bits/s
  64-chip temporal CDMA at 100 bits/s using a 12.8 kHz carrier

LANE B: MONDAY FREE-FOR-ALL / BATTLE MODE
Use these files only after the Thursday no-coding rule is no longer relevant:
  battle_transmitter_fsk.py
  battle_receiver_fsk.py
  battle_auto_test_fsk.py
  grand_demo_monday_only.py

The battle FSK branch uses repetition/interleaving/CRC style robustness. That is not for Thursday graded use. It exists because it is the one that survived your noisy room tests.

Run folders:
Every important script writes artifacts under ./runs/YYYYMMDD_HHMMSS_label_###.
That folder contains capture WAVs, JSON metrics, PNG plots, command manifests, and generated TX WAVs. Files no longer overwrite each other.

Read COMMANDS_ALL_CASES.txt before demo day.
