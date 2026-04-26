from __future__ import annotations

import importlib

mods = ["numpy", "sounddevice", "matplotlib", "scipy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"[ok] {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"[missing] {m}: {e}")

try:
    import sounddevice as sd
    print("\n[audio devices]")
    print(sd.query_devices())
except Exception as e:
    print(f"[audio] cannot query devices: {e}")
