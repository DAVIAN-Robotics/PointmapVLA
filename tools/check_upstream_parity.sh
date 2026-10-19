#!/usr/bin/env bash
# Verify that every file listed in UPSTREAM_IDENTICAL.txt is byte-identical to
# upstream starVLA at the commit this repository was derived from.
#
# PointmapVLA keeps upstream's layout so that its additions can be contributed
# back as new files. This check makes sure no unintended edits have crept into
# the shared upstream files.
#
# Usage: tools/check_upstream_parity.sh [path-to-upstream-checkout]
# If no path is given, the upstream commit is fetched from GitHub into a
# temporary directory.
set -euo pipefail

UPSTREAM_REPO="https://github.com/starVLA/starVLA.git"
UPSTREAM_COMMIT="2e5f239bc0b1661d7d556bdba5071f3041544cc6"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/UPSTREAM_IDENTICAL.txt"

if [[ $# -ge 1 ]]; then
  UPSTREAM_DIR="$1"
else
  UPSTREAM_DIR="$(mktemp -d)"
  trap 'rm -rf "$UPSTREAM_DIR"' EXIT
  git -C "$UPSTREAM_DIR" init -q
  git -C "$UPSTREAM_DIR" remote add origin "$UPSTREAM_REPO"
  git -C "$UPSTREAM_DIR" fetch -q --depth 1 origin "$UPSTREAM_COMMIT"
  git -C "$UPSTREAM_DIR" checkout -q FETCH_HEAD
fi

rc=0
while IFS= read -r f; do
  [[ -z "$f" || "$f" == \#* ]] && continue
  if [[ ! -e "$UPSTREAM_DIR/$f" ]]; then
    echo "MISSING UPSTREAM  $f"; rc=1; continue
  fi
  if [[ ! -e "$ROOT/$f" ]]; then
    echo "MISSING LOCAL     $f"; rc=1; continue
  fi
  if ! cmp -s "$UPSTREAM_DIR/$f" "$ROOT/$f"; then
    echo "DRIFT             $f"; rc=1
  fi
done < "$MANIFEST"

if [[ $rc -eq 0 ]]; then
  echo "OK: all $(grep -cvE '^\s*(#|$)' "$MANIFEST") manifest files match upstream $UPSTREAM_COMMIT"
fi
exit $rc
