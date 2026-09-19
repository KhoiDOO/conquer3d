#!/usr/bin/env bash
#
# Build and install conquer3d into the active conda environment.
#
# A bare `pip install .` picks the toolchain up from the system, which fails in two
# ways on a machine that also has a system CUDA install:
#
#   * PyTorch's extension builder falls back to /opt/cuda (or /usr/local/cuda), which
#     can be a different major version than the one PyTorch was built against. A CUDA
#     13 toolkit rejects sm_101 outright -- an architecture that existed only in 12.8
#     and 12.9 -- so the build dies with "Unsupported gpu architecture 'compute_101'".
#   * The system C++ compiler may be newer than the one the environment expects. GCC 16
#     fails on maths/ops.h with "'double rsqrt(double)' was declared 'extern' and later
#     'static'".
#
# This script pins the toolkit, the architecture list and the compilers to the active
# environment, so the build matches what PyTorch was compiled against.
#
# Usage:
#   conda activate <env>
#   ./install.sh                  # pip install . --no-build-isolation
#   ./install.sh -e               # extra arguments are forwarded to pip
#   ./install.sh -v --no-cache-dir
#   DRY_RUN=1 ./install.sh        # print the command and the resolved settings only
#
# Anything already exported wins, so a deliberate override is never clobbered.

set -euo pipefail

die() {
    printf 'install.sh: error: %s\n' "$*" >&2
    exit 1
}

note() {
    printf 'install.sh: %s\n' "$*"
}

# --- the active environment -------------------------------------------------------

[[ -n "${CONDA_PREFIX:-}" ]] ||
    die "no conda environment is active (CONDA_PREFIX is unset). Run 'conda activate <env>' first."

PY="$CONDA_PREFIX/bin/python"
[[ -x "$PY" ]] || die "no python interpreter at $PY"

# Build from the directory holding this script, wherever it is invoked from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TORCH_CUDA="$("$PY" - <<'EOF' 2>/dev/null || true
try:
    import torch
    print(torch.version.cuda or "")
except Exception:
    pass
EOF
)"

[[ -n "$TORCH_CUDA" ]] ||
    die "torch with CUDA support is not importable from $PY. --no-build-isolation builds against the active environment, so install the build requirements there first."

# --- CUDA toolkit -----------------------------------------------------------------

if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -x "$CONDA_PREFIX/bin/nvcc" ]]; then
        export CUDA_HOME="$CONDA_PREFIX"
    else
        note "warning: no nvcc inside the environment; leaving CUDA_HOME to PyTorch's own search, which may select a system toolkit."
    fi
fi

if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    NVCC_VERSION="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
    if [[ "${NVCC_VERSION%%.*}" != "${TORCH_CUDA%%.*}" ]]; then
        note "warning: nvcc is CUDA $NVCC_VERSION but torch was built against $TORCH_CUDA. Mismatched major versions frequently fail to link."
    fi
else
    NVCC_VERSION="(not found)"
fi

# --- target architectures ---------------------------------------------------------

# Without this, PyTorch asks for every architecture its toolkit knows, including ones a
# newer or older nvcc will refuse. The list matches setup.py: Turing through Hopper,
# plus PTX so future cards still run.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
fi

# --- compilers --------------------------------------------------------------------

# Conda ships its own GCC as <triplet>-linux-gnu-gcc; prefer it over whatever /usr/bin
# happens to hold, since the environment's headers were built against it.
if [[ -z "${CC:-}" ]]; then
    CONDA_CC="$(ls "$CONDA_PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null | head -n 1 || true)"
    [[ -n "$CONDA_CC" ]] && export CC="$CONDA_CC"
fi

if [[ -z "${CXX:-}" ]]; then
    CONDA_CXX="$(ls "$CONDA_PREFIX"/bin/*-linux-gnu-c++ 2>/dev/null | head -n 1 || true)"
    [[ -n "$CONDA_CXX" ]] && export CXX="$CONDA_CXX"
fi

# nvcc drives the host compiler itself, and warns if -ccbin is given twice.
if [[ -z "${NVCC_FLAGS:-}" && -n "${CXX:-}" ]]; then
    export NVCC_FLAGS="-ccbin ${CXX} -O3"
fi

# --- report and run ---------------------------------------------------------------

note "environment      $CONDA_PREFIX"
note "python           $("$PY" --version 2>&1)"
note "torch CUDA       $TORCH_CUDA"
note "CUDA_HOME        ${CUDA_HOME:-(PyTorch default)}"
note "nvcc             $NVCC_VERSION"
note "arch list        $TORCH_CUDA_ARCH_LIST"
note "CC               ${CC:-(system default)}"
note "CXX              ${CXX:-(system default)}"

CMD=("$PY" -m pip install . --no-build-isolation "$@")

if [[ -n "${DRY_RUN:-}" ]]; then
    note "dry run, nothing executed. Command would be:"
    printf '   '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

note "running: ${CMD[*]}"
"${CMD[@]}"
