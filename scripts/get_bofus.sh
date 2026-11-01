#!/usr/bin/env bash
# Fetch bofus, the code of Sartor et al. (2026), into third_party/bofus at the commit the
# experiments used. It provides the baselines, the benchmarks and the acquisition machinery
# (MIT license, in its own repository); pyproject.toml installs it from there.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/davidesartor/bofus.git
COMMIT=9d8976d8b408941dca6d950e1775d3013e54e81c   # branch `submission`
if [ -e third_party/bofus ] && [ ! -d third_party/bofus/.git ]; then
  echo "third_party/bofus exists and is not a git checkout; remove it first" >&2
  exit 1
fi
if [ ! -d third_party/bofus/.git ]; then
  mkdir -p third_party
  git clone --quiet "$URL" third_party/bofus
fi
git -C third_party/bofus fetch --quiet origin
git -C third_party/bofus checkout --quiet "$COMMIT"
echo "third_party/bofus at $(git -C third_party/bofus rev-parse --short HEAD)"
