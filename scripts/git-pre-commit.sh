#!/usr/bin/env bash
# Git pre-commit hook: format staged C++ files with clang-format.
# Install with:  ln -s ../../scripts/git-pre-commit.sh .git/hooks/pre-commit
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

command -v clang-format >/dev/null 2>&1 || { echo "error: clang-format not found in PATH" >&2; exit 1; }

mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMRTUXB \
    -- '*.cpp' '*.hpp' '*.h')

if [[ ${#files[@]} -eq 0 ]]; then
    exit 0
fi

formatted=()
for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    if ! clang-format --dry-run --Werror "$f" >/dev/null 2>&1; then
        clang-format -i "$f"
        formatted+=("$f")
    fi
done

if [[ ${#formatted[@]} -gt 0 ]]; then
    git add -- "${formatted[@]}"
    echo "clang-format: reformatted and re-staged:"
    printf '  %s\n' "${formatted[@]}"
fi
