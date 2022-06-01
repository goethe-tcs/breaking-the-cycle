#!/usr/bin/env bash

set -euov pipefail
cd "$(dirname "$0")"

OUTPUT='logs/kernels'
ARGS="-v --export-sccs --export-dfvs data/*/*/* -o $OUTPUT"

# remove old kernels
rm -R $OUTPUT || true

# reduce graphs
mkdir -p $OUTPUT
cargo +nightly run --features=cli --release --package dfvs --bin dfvs-preprocessing -- $ARGS
