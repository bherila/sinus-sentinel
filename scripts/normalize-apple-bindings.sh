#!/usr/bin/env bash
set -euo pipefail

for root in "$@"; do
  while IFS= read -r -d '' generated; do
    perl -pi -e 's/[ \t]+$//' "$generated"
    perl -0777 -pi -e 's/\n+\z/\n/' "$generated"
  done < <(find "$root" -type f \( -name '*.swift' -o -name '*.h' \) -print0)
done
