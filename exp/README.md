## Experiment naming conventions

- To make interacting with experiments easier for everyone please adhere to the conventions proposed in this file
:^)


- Create one directory `exp/<experiment-name>` for your experiment and use *kebab-case* for its name
(e.g.`exp/sim-anneal-params`)


- Try to stick to these four files for your experiment directory:
  - `exp/<experiment-name>/run.sh`, wich executes the experiment and can be executed from the projects root
  directory.
  - `exp/<experiment-name>/eval.py` (optional), which performs data post processing, like transforming output
  data and plotting.
  - `exp/<experiment-name>/job.sh` (optional), with appropriate SLURM settings to run your experiment on the
  Goethe-HLR.
  - Please make it as easy as feasible to execute these files and document any custom steps required to run your
    experiment in the `exp/<experiment-name>/README.md` (optional) file
  

- If your experiment requires a Rust binary that is tailored to the experiment:
  - Prefix its name with `exp` and use the name of the experiment directory for this `.rs` file
  (e.g. `src/bin/exp-sim-anneal-params.rs`)
  - Save temporary files that you create during the experiment in the `logs/<experiment-name>/` directory
