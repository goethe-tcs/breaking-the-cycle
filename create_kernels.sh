#!/usr/bin/env bash

set -euov pipefail
cd "$(dirname "$0")"

ARGS='-v --export-sccs data/*/*/*'
OUTPUT='data/kernels'

# remove old kernels
rm -R $OUTPUT || true

# reduce graphs
cargo +nightly run --features=cli,pace-digest --release --package dfvs --bin dfvs-preprocessing -- $ARGS

# move kernels to separate directory
mkdir $OUTPUT
rsync -am --remove-source-files --include='*/' --include='*_kernel.*' --exclude='*' --exclude=$OUTPUT data/ $OUTPUT
