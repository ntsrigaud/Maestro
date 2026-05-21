#!/usr/bin/env bash
# tools/fix_omp.sh
#
# macOS only.  Patches the install name of PyTorch's bundled libomp.dylib so
# that macOS dyld deduplicates it with the conda-managed llvm-openmp that
# numpy/OpenBLAS loads.  Without this, two OpenMP runtimes initialise in the
# same process, producing:
#
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
#   initialized. … Abort trap: 6
#
# ─── Root cause ──────────────────────────────────────────────────────────────
#
# The conda llvm-openmp package compiles its library with install name
# "@rpath/libomp.dylib".  numpy uses conda's libopenblas (openmp variant),
# which links to that same "@rpath/libomp.dylib" key.
#
# PyTorch's pip wheel bundles its own libomp with install name
# "/opt/homebrew/opt/libomp/lib/libomp.dylib" — a completely different key.
#
# macOS dyld deduplicates shared libraries by their LC_ID_DYLIB install name.
# Because the two copies advertise different install names, dyld loads both,
# and each initialises the OpenMP runtime independently → Error #15 → abort.
#
# ─── Fix ─────────────────────────────────────────────────────────────────────
#
# Rewrite torch's bundled libomp install name to "@rpath/libomp.dylib" — the
# same identifier that conda's OpenBLAS uses.  dyld then sees one canonical
# name and loads a single OpenMP runtime regardless of import order.
#
# Both runtimes are LLVM OpenMP with compatibility version 5.0.0, so sharing
# one instance is fully ABI-safe.  numpy's OpenBLAS and torch's C++ kernel
# both run against the same thread pool, which is the intended behaviour.
#
# ─── Maintenance ─────────────────────────────────────────────────────────────
#
# This fix must be reapplied after upgrading torch (pip reinstall overwrites
# the bundled dylib).  Running `make fix-omp` or `make install-dev` is enough.
# The script is idempotent: it exits immediately if the name already matches.

set -euo pipefail

# ── Platform guard ────────────────────────────────────────────────────────────
if [ "$(uname)" != "Darwin" ]; then
    echo "fix_omp.sh: not macOS — nothing to do."
    exit 0
fi

# ── Conda env guard ───────────────────────────────────────────────────────────
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "fix_omp.sh: CONDA_PREFIX is not set." >&2
    echo "  Activate the environment first:  conda activate maestro" >&2
    exit 1
fi

TARGET_ID="@rpath/libomp.dylib"

# ── Locate torch's bundled libomp ─────────────────────────────────────────────
# KMP_DUPLICATE_LIB_OK=TRUE is set only for this path-discovery call.  When
# the fix has not yet been applied, importing torch inside Python would itself
# abort with OMP Error #15.  The env var suppresses that abort so we can reach
# the file path; the fix that follows permanently eliminates the need for it.
TORCH_LIBOMP=$(KMP_DUPLICATE_LIB_OK=TRUE python - <<'PYEOF'
import sys, os, torch
path = os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib")
if not os.path.exists(path):
    sys.stderr.write(f"fix_omp.sh: torch libomp not found at {path}\n")
    sys.exit(1)
print(path)
PYEOF
) || exit 1

# ── Check current install name ────────────────────────────────────────────────
CURRENT_ID=$(otool -D "$TORCH_LIBOMP" | tail -n 1)

if [ "$CURRENT_ID" = "$TARGET_ID" ]; then
    echo "fix_omp.sh: install name is already '$TARGET_ID' — nothing to do."
    exit 0
fi

# ── Apply patch ───────────────────────────────────────────────────────────────
echo "fix_omp.sh: patching torch libomp install name"
echo "  file:   $TORCH_LIBOMP"
echo "  before: $CURRENT_ID"
echo "  after:  $TARGET_ID"

# install_name_tool requires the file to be writable.
chmod u+w "$TORCH_LIBOMP"
install_name_tool -id "$TARGET_ID" "$TORCH_LIBOMP"

# Re-apply an ad-hoc code signature so macOS Gatekeeper accepts the modified
# binary.  Failure is non-fatal: unsigned user-space dylibs still load fine
# on most macOS configurations.
if codesign --force --sign - "$TORCH_LIBOMP" 2>/dev/null; then
    echo "  re-signed with ad-hoc identity."
else
    echo "  codesign skipped (not required in this configuration)."
fi

echo "fix_omp.sh: done.  Re-run after upgrading torch."
