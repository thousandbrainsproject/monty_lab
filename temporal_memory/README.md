# Temporal Memory for Object Recognition

## Structure

`.py` files in the base folder contain working code for TM (data processing, SDR creation, training, inference, visualization, etc.). The `notebooks/` folder contains experimental notebooks for training, validation of data, and SDR creation. Use these notebooks with caution. The `old/` folder contains old code that validated old experiments. Some of this code might be deprecated.

## Overview

Please run `python run.py -e <experiment name>`. Experiment configs can be found in `experiments/` -- please edit these configs as needed to run any custom experiments.
The `run.py` script will fail if the processed dataset (curvature and location SDRs, etc.) is not found. If it is not found, then follow the script's instructions
to generate this dataset (with the help of scripts like `process_data.py`, `coordinate_encoder.py`, `curvature_encoder.py`, and/or `cluster.py`).

## Tutorial

### Data generation

Test to see if the conda installation is successful by checking to see if the following `import` works:

```python```

```from nupic.bindings.math import SparseMatrixConnections```

Ensure that trimesh is the correct version: `pip install trimesh==3.10.8`. Later versions of trimesh don't always produce the correct results for these experiments.

Flags:

`sdr_p`: where to save the dataset

`ycb_p`: root directory of the YCB path (most likely `~/tbp/data/habitat/objects/ycb`)

`-r`: hash radius in Cartesian Coordinate space (ultimately determines how much one SDR overlaps with another)

`-d1`, `-d2`: *range* of objects to save

`-n`: either the total number of bits in the SDR (in the case of the `curvature_encoder.py` and `coordinate_encoder.py`) or the total number of training points to generate (in the case of `cluster.py`)

`-w`: total number of *ON* bits in the SDR

`coord`: wheteher to cluster datapoints by coordinates -- `n` most commonly occurring clusters are chosen for training

`curve`: whether to cluster datapoints by curvatures -- `n` most commonly occurring clusters are chosen for training


Pre-process the data:

Update `<YOU>` in the following lines to reflect your user ID. The following are single-line commands.

```python ~/tbp/tbp.monty/projects/temporal_memory/process_data.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/temporal_memory/tm_dataset -ycb_p ~/tbp/data/habitat/objects/ycb```

```python ~/tbp/tbp.monty/projects/temporal_memory/curvature_encoder.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/temporal_memory/tm_dataset -r 5 -d1 0 -d2 10 -n 1024 -w 11```

```python ~/tbp/tbp.monty/projects/temporal_memory/coordinate_encoder.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/temporal_memory/tm_dataset -r 5 -d1 0 -d2 10 -n 2048 -w 30```

```python ~/tbp/tbp.monty/projects/temporal_memory/cluster.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/temporal_memory/tm_dataset -n 50 -coord True -curve True```

### Run the experiment

`python run.py -e <CONFIG NAME>`. For example, you can run `python run.py -e occlusion_cluster_by_coord_curve`.