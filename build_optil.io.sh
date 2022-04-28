#!/usr/bin/env bash
set -e
RUSTFLAGS='-C target-feature=+crt-static -C target-cpu=haswell' cargo build --release --target x86_64-unknown-linux-gnu --bin optil_exact