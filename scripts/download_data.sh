#!/usr/bin/env bash
# Download the released data (episode logs, training corpus, AgentDyn copy) and unpack it into data/.
# The archive is attached to the GitHub release because it is larger than GitHub's limit for repository files.
set -e; cd "$(dirname "$0")/.."
URL=https://github.com/annonyMss/caval/releases/download/v1.0.0.1/caval_data.tar.gz
SHA256=67574596bea7f5c57ce5bfc3c4b60cad518a714e63e67652ca493f2957e16f6f
if [ -d data/runs ] && [ -d data/corpus ]; then echo "data/ is already present"; exit 0; fi
echo "downloading caval_data.tar.gz (237 MB) ..."
curl -L --fail --progress-bar -o caval_data.tar.gz "$URL"
echo "$SHA256  caval_data.tar.gz" | sha256sum --check --status || { echo "checksum mismatch, download again"; rm -f caval_data.tar.gz; exit 1; }
tar -xzf caval_data.tar.gz && rm caval_data.tar.gz
echo "data/ ready: $(find data -type f | wc -l) files"
