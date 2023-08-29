# The Pivoting Framework: Frank-Wolfe Algorithms with Active Set Size Control

## References

This is the accompanying repository for the paper:

> Wirth, E., Besançon, M., and Pokutta, S. (2023). The Pivoting Framework: Frank-Wolfe Algorithms with Active Set Size Control.

## Installation guide

The repository is a standard Julia package, open Julia in the repository and run:

```julia
import Pkg
Pkg.activate(".")
Pkg.update()

import PivotingFrankWolfe
```

See the [Julia documentation](https://docs.julialang.org/en/v1/stdlib/Pkg/) for more details on working with the Julia package manager.

## Adding experiment data

Due to file sizes, the largest experiment datasets are not added to the repository.
They can be found at the following location and added in the corresponding folders:

- **Logistic regression**: the validation set used in the paper is already present. The training set is available on the [UCI ML repository](https://archive.ics.uci.edu/ml/datasets/Gisette) and can be added to `experiments/GISETTE`.

## Running the paper experiments

The experiments are scripts are run with:

```bash
julia --project experiments/run_logreg.jl
julia --project experiments/run_birkhoff.jl
julia --project experiments/run_signal_recovery.jl
```

This will populate the `experiments/results` folder with the result JSON files.
One can then run:

```bash
julia --project experiments/plot_results.jl
```
which produces the sparsity and trajectory plots for each experiment in `experiments/plots`.
