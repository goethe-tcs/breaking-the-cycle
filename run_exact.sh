#!/usr/bin/env bash

cd "$(dirname "$0")"

DIR=data/pace/exact_public
rm -f DIR/*.solution

cargo b --all-features --release --bin dfvs-cli

ls -1 $DIR/e_*.metis | \
  parallel --timeout 600 \
    target/release/dfvs-cli -vv -m exact -i {} -o {.}.solution

for f in $DIR/*.solution
do
  g=$(echo $f | perl -p -e 's/.solution//')
  echo $g
  cat $f | wc -l
  verifier/verifier $g $f
done
