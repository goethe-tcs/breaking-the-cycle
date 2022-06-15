# BreakingTheCycle
This repository contains exact and heuristic solvers for the **Directed Feedback Vertex Set (DFVS)** problem. 
They were developed as a submission for the [PACE 2022 Challenge](https://pacechallenge.org/2022).

# Installing the Rust ecosystem
The solvers are implemented in Rust.
For information on the installation of the compiler and the default package manager `cargo` we refer to the [official documentation](https://www.rust-lang.org/tools/install).
(At time of writing) we require language features that are only available in the `nightly channel` of the Rust universe; see [nightly channel](https://rust-lang.github.io/rustup/concepts/channels.html) for details.
On most Unix-like systems the whole installation boils down to:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```
(use the *piping to s(hell)* idiom at your own discretion)

All further dependencies are handled by `cargo` package manager.

# Building for PACE
For PACE-related benchmarking we provide the script `build_optil.io.sh` which configures the compiler to emit code optimized for the OPTIL enviroment.
Amongst others, it enables AVX2 and BMI2 for our base case solver.
Observe that `BMI2` is also available on AMD machines but too slow; for these machines we provide a faster software implementation and ask to build without the BMI2 support (by modifying the script accordingly).
The script compiles the two binaries `target/x86_64-unknown-linux-gnu/release/{optil_exact,optil_heuristic}` which adhere to the I/O-model prescribed by [OPTIL.io](https://www.optil.io).

```bash
./build_optil.io.sh
cat {YOUR-METIS-FILE} | target/x86_64-unknown-linux-gnu/release/optil_exact > exact_solution
cat {YOUR-METIS-FILE} | target/x86_64-unknown-linux-gnu/release/optil_heuristic > a_solution
```

# A more convient way
If you want to use the exact solver outside of PACE it might be more convenient to interact with it using the more versatile `dfvs-cli` interface.
To build it, run:
**Important**:
```bash
cargo build --release --all-features
```

This places the binary into the folder `target/release`.
The solver accepts files in the `Metis` format as described [here](https://pacechallenge.org/2022/tracks/):
```
target/release/dfvs-cli -i {YOUR-METIS-FILE} -o {YOUR-SOLUTION-FILE}
```

See also `dfvs-cli --help` for further options.

# Solver Descriptions
There are also two solver descriptions available, which offer a short summary of our solutions for the [exact track](https://github.com/goethe-tcs/breaking-the-cycle/blob/master/exact.pdf) and [heuristic track](https://github.com/goethe-tcs/breaking-the-cycle/blob/master/heuristic.pdf).
