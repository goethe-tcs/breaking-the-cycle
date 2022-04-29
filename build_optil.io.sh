#!/usr/bin/env bash
set -e

export RUSTFLAGS='-C target-feature=+crt-static -C target-cpu=haswell'
BUILD_CMD="cargo build --release --target x86_64-unknown-linux-gnu"

$BUILD_CMD --bin optil_exact
$BUILD_CMD --bin optil_heuristic --features="signal-handling"
