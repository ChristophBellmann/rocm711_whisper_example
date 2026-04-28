#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm}"
WHEEL_DIR="${WHEEL_DIR:-${ROCM_PREFIX}/wheels/pytorch_rocm711}"
TORCH_WHEEL="${TORCH_WHEEL:-}"

usage() {
  cat <<'USAGE'
Usage: ./setup_rocm_venv.sh [options]

Creates/updates a Python venv for the installed custom ROCm PyTorch stack.

Options:
  --venv <dir>         Venv directory (default: .venv)
  --rocm-prefix <dir>  Installed ROCm prefix (default: /opt/rocm)
  --torch-wheel <path> Explicit torch wheel (default: torch-current.whl or newest torch-*.whl)
  -h, --help           Show help
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --rocm-prefix)
      ROCM_PREFIX="${2:-}"
      WHEEL_DIR="${ROCM_PREFIX}/wheels/pytorch_rocm711"
      shift 2
      ;;
    --torch-wheel)
      TORCH_WHEEL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown arg: $1"
      ;;
  esac
done

command -v python3 >/dev/null 2>&1 || die "python3 not found"
[[ -d "${ROCM_PREFIX}" ]] || die "ROCm prefix not found: ${ROCM_PREFIX}"

if [[ -z "${TORCH_WHEEL}" ]]; then
  if [[ -f "${WHEEL_DIR}/torch-current.whl" ]]; then
    TORCH_WHEEL="${WHEEL_DIR}/torch-current.whl"
  else
    TORCH_WHEEL="$(ls -1t "${WHEEL_DIR}"/torch-*.whl 2>/dev/null | head -n 1 || true)"
  fi
fi

[[ -n "${TORCH_WHEEL}" ]] || die "No torch wheel found in ${WHEEL_DIR}"
[[ -f "${TORCH_WHEEL}" ]] || die "Torch wheel not found: ${TORCH_WHEEL}"
if [[ -L "${TORCH_WHEEL}" ]]; then
  TORCH_WHEEL="$(readlink -f "${TORCH_WHEEL}")"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install --upgrade --force-reinstall "${TORCH_WHEEL}"

activate_script="${VENV_DIR}/bin/activate_rocm_pytorch.sh"
python_wrapper="${VENV_DIR}/bin/python-rocm"

cat >"${activate_script}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")/.." && pwd)"
export ROCM_PATH="${ROCM_PREFIX}"
export HIP_PATH="\${HIP_PATH:-\$ROCM_PATH}"
export HSA_PATH="\${HSA_PATH:-\$ROCM_PATH}"
export PATH="\$ROCM_PATH/bin:\$ROCM_PATH/llvm/bin:\$VENV_DIR/bin:\${PATH:-}"
export LD_LIBRARY_PATH="\$ROCM_PATH/lib:\$ROCM_PATH/lib64:\$ROCM_PATH/lib/llvm/lib:\$ROCM_PATH/lib/host-math/lib:\$ROCM_PATH/lib/rocm_sysdeps/lib:\$ROCM_PATH/llvm/lib:\${LD_LIBRARY_PATH:-}"
if [[ -f "\$ROCM_PATH/lib/llvm/lib/libomp.so" ]]; then
  export LD_PRELOAD="\$ROCM_PATH/lib/llvm/lib/libomp.so\${LD_PRELOAD:+:\${LD_PRELOAD}}"
fi
export USE_ROCM_HIPBLASLT="\${USE_ROCM_HIPBLASLT:-0}"
if [[ -z "\${HIP_DEVICE_LIB_PATH:-}" ]]; then
  if [[ -d "\$ROCM_PATH/lib/llvm/amdgcn/bitcode" ]]; then
    export HIP_DEVICE_LIB_PATH="\$ROCM_PATH/lib/llvm/amdgcn/bitcode"
  elif [[ -d "\$ROCM_PATH/amdgcn/bitcode" ]]; then
    export HIP_DEVICE_LIB_PATH="\$ROCM_PATH/amdgcn/bitcode"
  fi
fi

# shellcheck source=/dev/null
source "\$VENV_DIR/bin/activate"
SCRIPT
chmod +x "${activate_script}"

cat >"${python_wrapper}" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_rocm_pytorch.sh"
exec "${VIRTUAL_ENV}/bin/python" "$@"
SCRIPT
chmod +x "${python_wrapper}"

echo "Installed ${TORCH_WHEEL}"
echo "Activate with:"
echo "  source \"${activate_script}\""
