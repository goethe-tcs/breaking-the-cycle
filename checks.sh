#!/usr/bin/env bash

set -euov pipefail

cd "$(dirname "$0")"
FEATURES="cli,pace-logging"

if ! cargo +nightly fmt --check; then
  echo "Run 'cargo fmt' to auto-format the sources or apply previous changes manually <--------------------------------------"
fi

cargo +nightly clippy --features="$FEATURES" -- -D clippy::all

if [ "$(uname -m)" == "x86_64" ]; then
  cargo +nightly tarpaulin --features="$FEATURES" -o html
else
  cargo +nightly test
fi