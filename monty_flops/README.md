# Computing FLOPs in Monty

Using `pypapi` to compute FLOPs in Monty.
Note that `pypapi` only works on Linux (Oracle Node).

1. Create a new conda environment in Linux (Oracle Node).

```bash
conda create -n monty_flops python=3.9
conda activate monty_flops
```

2. Install `pypapi`.

```bash
pip install python-papi
```
