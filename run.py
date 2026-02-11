#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
import time
import wave
from pathlib import Path


def _fmt_s(x: float) -> str:
    if x < 1e-3:
        return f"{x * 1e6:.0f}us"
    if x < 1.0:
        return f"{x * 1e3:.1f}ms"
    return f"{x:.3f}s"


def _maybe_reexec_with_rocm_env() -> None:
    """
    Ensure ROCm runtime libs are discoverable for the process.

    This ROCm 7.11 torch wheel may require OpenMP runtime symbols (__kmpc_*).
    Preloading libomp satisfies that at import time.
    """
    if os.environ.get("ROCM711_EXAMPLE_REEXEC", "") == "1":
        return

    rocm = os.environ.get("ROCM_PATH", "").strip() or "/opt/rocm"
    env = dict(os.environ)
    env["ROCM_PATH"] = rocm

    # PATH: rocminfo/hip tools if needed (not required for whisper itself).
    path = env.get("PATH", "")
    want_path = [f"{rocm}/bin", f"{rocm}/llvm/bin"]
    if not all(p in path.split(":") for p in want_path):
        env["PATH"] = ":".join(want_path + ([path] if path else []))

    # LD_LIBRARY_PATH: runtime libs for ROCm.
    ld = env.get("LD_LIBRARY_PATH", "")
    want_ld = [
        f"{rocm}/lib",
        f"{rocm}/lib64",
        f"{rocm}/lib/llvm/lib",
        f"{rocm}/lib/host-math/lib",
        f"{rocm}/lib/rocm_sysdeps/lib",
        f"{rocm}/llvm/lib",
    ]
    if not all(p in ld.split(":") for p in want_ld):
        env["LD_LIBRARY_PATH"] = ":".join(want_ld + ([ld] if ld else []))

    # LD_PRELOAD: libomp for __kmpc_*.
    libomp = f"{rocm}/lib/llvm/lib/libomp.so"
    cur = env.get("LD_PRELOAD", "")
    if os.path.exists(libomp) and libomp not in cur.split(":"):
        env["LD_PRELOAD"] = ":".join([libomp] + ([cur] if cur else []))

    if (
        env.get("PATH") != os.environ.get("PATH")
        or env.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH")
        or env.get("LD_PRELOAD") != os.environ.get("LD_PRELOAD")
    ):
        env["ROCM711_EXAMPLE_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, __file__] + sys.argv[1:], env)


def _write_test_wav(path: Path, *, seconds: int, sr: int = 16000, freq_hz: float = 440.0) -> None:
    """
    Deterministic PCM16 mono WAV: a simple tone with a slow amplitude envelope.

    This is not speech; it's just a stable audio input so the example is self-contained.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = seconds * sr

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)

        # Stream in chunks so we don't allocate huge buffers.
        chunk = 8192
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            frames = bytearray()
            for i in range(i0, i1):
                t = i / sr
                # Envelope: 0..1..0 over the whole clip.
                env = 0.5 * (1.0 - math.cos(2.0 * math.pi * (t / max(1e-9, seconds))))
                x = env * math.sin(2.0 * math.pi * freq_hz * t)
                s = int(max(-1.0, min(1.0, x)) * 32767.0)
                frames += int(s).to_bytes(2, byteorder="little", signed=True)
            wf.writeframes(frames)


def main() -> int:
    _maybe_reexec_with_rocm_env()

    try:
        import torch  # type: ignore
    except Exception as e:
        print("FAIL: import torch")
        print(f"  {e!r}")
        return 1

    try:
        import whisper  # type: ignore
    except Exception as e:
        print("FAIL: import whisper")
        print(f"  {e!r}")
        return 1

    rocm = os.environ.get("ROCM_PATH", "").strip() or "/opt/rocm"
    model_name = (os.environ.get("WHISPER_MODEL", "") or "tiny.en").strip()
    audio_target_s = int(float(os.environ.get("AUDIO_TARGET_S", "300") or "300"))

    print("== Whisper ROCm example ==")
    print(f"torch.__version__       : {torch.__version__}")
    print(f"torch.version.hip       : {getattr(torch.version, 'hip', None)}")
    print(f"torch.version.rocm      : {getattr(torch.version, 'rocm', None)}")
    print(f"ROCM_PATH               : {rocm}")
    print(f"torch.cuda.is_available : {bool(torch.cuda.is_available())}")

    if not torch.cuda.is_available():
        print("FAIL: HIP device not available to PyTorch (torch.cuda.is_available=false).")
        return 2

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    print(f"device                 : {props.name}")
    print(f"gcnArchName (if any)   : {getattr(props, 'gcnArchName', None)}")

    work = Path(__file__).resolve().parent / "work"
    audio = work / f"tone_{audio_target_s}s_{model_name.replace('/', '_')}.wav"
    if not audio.exists() or audio.stat().st_size == 0:
        print(f"generating audio        : {audio.name} ({audio_target_s}s)")
        _write_test_wav(audio, seconds=audio_target_s)

    print(f"model                  : {model_name}")
    print(f"audio                  : {audio.name}")

    # Workaround: load checkpoint on CPU, then move model to GPU.
    t0 = time.perf_counter()
    model = whisper.load_model(model_name, device="cpu").to("cuda")
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    print(f"model load+to(cuda)     : {_fmt_s(t_load)}")

    # One sustained run (whisper internally uses torch ops).
    # fp16=True first (faster), fallback to fp16=False.
    t1 = time.perf_counter()
    try:
        result = model.transcribe(
            str(audio),
            fp16=True,
            beam_size=5,
            best_of=5,
            language="en",
            task="transcribe",
            condition_on_previous_text=False,
        )
    except Exception:
        result = model.transcribe(
            str(audio),
            fp16=False,
            beam_size=5,
            best_of=5,
            language="en",
            task="transcribe",
            condition_on_previous_text=False,
        )
    torch.cuda.synchronize()
    t_run = time.perf_counter() - t1

    text = (result.get("text") or "").strip()
    words = len(text.split()) if text else 0
    wps = (words / t_run) if t_run > 0 else 0.0
    rtf = (t_run / float(audio_target_s)) if audio_target_s > 0 else 0.0

    print("")
    print("== Results ==")
    print(f"wall                   : {_fmt_s(t_run)}")
    print(f"audio_s                : {audio_target_s}")
    print(f"rtf (wall/audio)       : {rtf:.3f}")
    print(f"words                  : {words}")
    print(f"words/s                : {wps:.2f}")
    print("")
    print("text (first 200 chars):")
    print(text[:200])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

