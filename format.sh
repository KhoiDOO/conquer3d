#!/usr/bin/env bash
# ==============================================================================
# Conquer3D - Code Formatting Script (clang-format)
# Formats all C, C++, and CUDA source & header files in the conquer3d codebase.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

# Defaults
CHECK_MODE=false
VERBOSE=false
TARGET_DIR="$ROOT_DIR/conquer3d"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [DIRECTORY]

Formats all C, C++, and CUDA files (*.cpp, *.cu, *.cuh, *.h, *.hpp, *.c)
using clang-format according to the repository's .clang-format configuration.

Arguments:
  DIRECTORY          Target directory to scan (default: conquer3d)

Options:
  -c, --check        Check formatting without modifying files (exits with error if unformatted)
  -n, --dry-run      Alias for --check
  -v, --verbose      List each file as it is processed
  -h, --help         Show this help message and exit

Examples:
  $(basename "$0")                     # Format all C++/CUDA files in conquer3d in-place
  $(basename "$0") --check             # Check formatting across all files
  $(basename "$0") conquer3d/csrc/ops  # Format only ops directory
EOF
}

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--check|--dry-run|-n)
            CHECK_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option '$1'" >&2
            usage
            exit 1
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

# Resolve TARGET_DIR to absolute path if relative
if [[ ! "$TARGET_DIR" = /* ]]; then
    TARGET_DIR="$ROOT_DIR/$TARGET_DIR"
fi

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Target directory '$TARGET_DIR' does not exist." >&2
    exit 1
fi

# Verify clang-format binary
if ! command -v clang-format &>/dev/null; then
    echo "Error: 'clang-format' is not installed or not found in PATH." >&2
    echo "Please install clang-format (e.g. 'sudo apt install clang-format' or conda)." >&2
    exit 1
fi

CLANG_FORMAT_VERSION="$(clang-format --version 2>/dev/null | head -n 1)"
echo "Using: $CLANG_FORMAT_VERSION"
echo "Target directory: $TARGET_DIR"

# Check for .clang-format configuration
if [[ ! -f "$ROOT_DIR/.clang-format" ]]; then
    echo "Warning: No .clang-format file found at '$ROOT_DIR/.clang-format'." >&2
fi

# Query all C/C++/CUDA coding files
echo "Scanning for C/C++/CUDA files..."
mapfile -t FILES < <(find "$TARGET_DIR" -type f \( \
    -name "*.cpp" -o \
    -name "*.cu"  -o \
    -name "*.cuh" -o \
    -name "*.h"   -o \
    -name "*.hpp" -o \
    -name "*.c"   -o \
    -name "*.cc"  -o \
    -name "*.cxx" \
\) | sort)

TOTAL_FILES=${#FILES[@]}
if [[ "$TOTAL_FILES" -eq 0 ]]; then
    echo "No C/C++/CUDA files found in '$TARGET_DIR'."
    exit 0
fi

echo "Found $TOTAL_FILES coding file(s)."

VIOLATIONS=0
PROCESSED=0

if [[ "$CHECK_MODE" = true ]]; then
    echo "Running formatting check (dry-run)..."
    for file in "${FILES[@]}"; do
        REL_PATH="${file#$ROOT_DIR/}"
        if ! clang-format --dry-run --Werror "$file" &>/dev/null; then
            echo "  [FAIL] $REL_PATH requires formatting"
            VIOLATIONS=$((VIOLATIONS + 1))
        elif [[ "$VERBOSE" = true ]]; then
            echo "  [OK]   $REL_PATH"
        fi
        PROCESSED=$((PROCESSED + 1))
    done

    echo "----------------------------------------------------"
    if [[ "$VIOLATIONS" -gt 0 ]]; then
        echo "Check failed: $VIOLATIONS file(s) require formatting out of $TOTAL_FILES checked." >&2
        echo "Run './format.sh' to reformat all files automatically." >&2
        exit 1
    else
        echo "Check passed: All $TOTAL_FILES file(s) are properly formatted."
        exit 0
    fi
else
    echo "Formatting files in-place..."
    for file in "${FILES[@]}"; do
        REL_PATH="${file#$ROOT_DIR/}"
        if [[ "$VERBOSE" = true ]]; then
            echo "  [FORMAT] $REL_PATH"
        fi
        clang-format -i "$file"
        PROCESSED=$((PROCESSED + 1))
    done

    echo "----------------------------------------------------"
    echo "Successfully formatted $PROCESSED file(s) in '$TARGET_DIR'."
fi
