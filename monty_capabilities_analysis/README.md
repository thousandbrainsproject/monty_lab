# Monty Capabilities Analysis

This repository contains the code for the Monty Capabilities Analysis. It has
the following structure:

```yaml
|-- monty_capabilities_analysis/
|-- notebooks/
|-- out/
|-- results/
|   |-- 20241104_benchmark_v9/
|   |-- dmc/
|-- scripts/

```

The `monty_capabilities_analysis` folder is a small package that contains common paths definitions and various data handling and plotting utilities. It'll be need to be added
to your python path to run some of the scripts.

Please create the `results` directory and place the `dmc` results folder into it. These
are copied from Scott's folder in Oracle Node. Alternatively, soft link to the
equivalent directory in your local machine.

`out` is a writable directory for saving plots and tables. Both `out` and `results`
will be ignored by git.

