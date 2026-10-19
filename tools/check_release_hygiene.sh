#!/usr/bin/env bash
# Fail if the tree contains internal lab artifacts: experiment IDs, run tags,
# internal code names, private filesystem paths, user names, or non-English
# (Hangul) notes. Everything in this repository must read as written for
# outside users.
#
# Usage: tools/check_release_hygiene.sh
set -euo pipefail
export LC_ALL=C.UTF-8   # needed for the Hangul code-point range below

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Patterns that must not appear anywhere in tracked text files. The first author's
# public homepage and project page (godnpeter.github.io) are allowed.
PATTERN='\bexp[0-9]{2}\b|\bp4[19]_|\bB2[0-9]\b|\bs2[567]_ee|pm_tower|XEMB|pm_mix17|made_by_claude|/home/nas_main|dongjinkim|dongyoonhwang|byungkunlee|godnpeter(?!\.github\.io)|[\x{AC00}-\x{D7A3}]'

# Files that legitimately mention upstream SHAs or license holders are fine;
# binary fixtures are skipped by -I.
matches=$(grep -rnIP --exclude-dir=.git --exclude-dir=.omc --exclude-dir=fixtures --exclude-dir=results --exclude-dir=playground --exclude-dir=third_party --exclude-dir=release_staging --exclude='check_release_hygiene.sh' --exclude='slurm-*.out' "$PATTERN" . || true)

if [[ -n "$matches" ]]; then
  echo "$matches"
  echo
  echo "FAIL: $(echo "$matches" | wc -l) line(s) contain internal artifacts"
  exit 1
fi
echo "OK: no internal artifacts found"
