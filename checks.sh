#!/bin/bash
cd `dirname $0`
set -v

if ! cargo fmt --check; then
  echo "Run 'cargo fmt' to auto-format the sources or apply previous changes manually <--------------------------------------"
fi

cargo clippy --all-features -- -D clippy::all

cargo test -q
