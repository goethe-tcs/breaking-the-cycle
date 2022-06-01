#!/usr/bin/env bash
set -e

export RUSTFLAGS='-C target-feature=+crt-static -C target-cpu=haswell -C target-feature=+bmi2,+bmi1'
BUILD_CMD="cargo +nightly build --release --target x86_64-unknown-linux-gnu"

$BUILD_CMD --bin optil_exact
$BUILD_CMD --bin optil_heuristic --features="signal-handling"
