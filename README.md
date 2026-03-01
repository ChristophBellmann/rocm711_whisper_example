# rocm711_whisper_example

Minimal Whisper example that:

- uses your **custom ROCm 7.11 PyTorch wheel** from `/opt/rocm/wheels/pytorch_rocm711/` (always local)
- installs Whisper (`openai-whisper`) from the internet (PyPI)
- generates a deterministic test WAV (no external audio needed)
- runs Whisper on the **GPU** and prints basic throughput metrics

## Prereqs

- `/opt/rocm` installed and working (`/opt/rocm/bin/rocminfo`)
- The custom torch wheel exists:
  - `/opt/rocm/wheels/pytorch_rocm711/torch-2.11.0a0+devrocm20260301-cp312-cp312-linux_x86_64.whl`
- Local custom ROCm Python index exists:
  - `/media/christoph/some_space/Compute/TheRock_gfx1031/build-stage2/python_packages_gfx1031/dist/simple/`

## Setup

```bash
cd /media/christoph/some_space/Compute/rocm711_whisper_example
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Run (default)

```bash
source .venv/bin/activate
python run.py
```

Defaults:
- model: `tiny.en`
- generated audio length: `300s`

Override (example):
```bash
WHISPER_MODEL=base AUDIO_TARGET_S=60 python run.py
```

## Notes

- First run downloads the Whisper model into your cache (expected).
- The script loads the model checkpoint on **CPU first** and then moves it to CUDA/HIP.
  This avoids crashes seen when deserializing directly to GPU on some ROCm/torch combos.
