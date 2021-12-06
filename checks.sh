#!/bin/bash
cd `dirname $0`
set -e
set -v
FEATURES="cli,pace-logging"

if ! cargo +nightly fmt --check; then
  echo "Run 'cargo fmt' to auto-format the sources or apply previous changes manually <--------------------------------------"
fi

cargo +nightly clippy --features=$FEATURES -- -D clippy::all

cargo +nightly tarpaulin --features=$FEATURES -o html
