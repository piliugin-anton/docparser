#!/usr/bin/env bash
# Source before building or running docparser-api with --features mkl.
# Adds MKL and libiomp5 (Intel OpenMP) to LD_LIBRARY_PATH.

set -eo pipefail

MKL_VARS="${MKL_VARS:-/opt/intel/oneapi/mkl/latest/env/vars.sh}"
COMPILER_VARS="${COMPILER_VARS:-/opt/intel/oneapi/compiler/latest/env/vars.sh}"

if [[ ! -f "$MKL_VARS" ]]; then
  echo "mkl-env: MKL vars not found at $MKL_VARS" >&2
  echo "Install intel-oneapi-mkl-devel or set MKL_VARS." >&2
  return 1 2>/dev/null || exit 1
fi

# Intel vars.sh scripts reference unset variables; avoid `set -u` while sourcing.
set +u
# shellcheck source=/dev/null
source "$MKL_VARS"

if [[ -f "$COMPILER_VARS" ]]; then
  # shellcheck source=/dev/null
  source "$COMPILER_VARS"
else
  echo "mkl-env: compiler vars not found at $COMPILER_VARS (libiomp5 may be missing at runtime)" >&2
  echo "Install intel-oneapi-openmp or set COMPILER_VARS." >&2
fi
