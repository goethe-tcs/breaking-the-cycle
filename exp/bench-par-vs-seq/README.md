## Compare parallel and sequentiell benchmarking

This experiment aims to analyze negative effects on the measured runtime of individual designpoints of an
experiments, when it's designpoints are run in parallel.
It is designed to be run on the `general1` partition of the Goethe-HLR computer cluster.

1. The `job.sh` file contains appropriate SLURM settings and can be queued with `sbatch job.sh`.

2. Run the `eval.py` file after all jobs of the job array finished by passing in the job directory
that is created in the `logs/exp-bench-par-vs-seq/` directory. You can also pass in multiple job directories to
compare the results of them.

You can also run this experiment on a different system, but you should adjust the tested thread counts to match
the physical CPU cores of that system. You can also run the `run.sh` file directly if your system doesn't support
SLURM, but you might want to increase the `ITERATIONS` variable to `20`.
