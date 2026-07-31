#!/usr/bin/env bash
# Verify all C++ sources are clang-formatted. Exit non-zero if any file differs.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

command -v clang-format >/dev/null 2>&1 || { echo "error: clang-format not found in PATH" >&2; exit 1; }

mapfile -t files < <(find include src tests examples \
    -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \
    | grep -v '/build/' | grep -v '/.cxx/' | sort)

clang-format --dry-run --Werror "${files[@]}"
echo "check-format: OK ($((${#files[@]})) files)"
