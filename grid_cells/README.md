# Grid cells for 3D Object Recall

## Structure

`.py` files in the base folder contain working code for Grid cells (data processing and SDR creation). The `notebooks/` folder contains experimental files and notebooks for exploring grid cell algorithms, training, inference, and validation of data. Use these notebooks with caution.

## Overview

All the core code for Grid cells have been moved to the Monty frameworks. The main experiment class that uses grid cells is in [`frameworks/experiments/htm_experiment.py`](/src/tbp/monty/frameworks/experiments/htm_experiment.py). The separate L4 and L6a (i.e. TM and grid cell layer) algorithms can be found in [`frameworks/models/htm.py`](/src/tbp/monty/frameworks/models/htm.py). The Learning Module that uses both of these layers and creates a joint L4 / L6a algorithm can be found in [`frameworks/models/htm_learning_modules.py`](/src/tbp/monty/frameworks/models/htm_learning_modules.py).

Please run ```python tbp.monty/projects/monty_runs/run.py -e <experiment name>```. Experiment configs can be found here in [`projects/monty_runs/experiments/htm.py`](/projects/monty_runs/experiments/htm.py) -- please edit these configs as needed to run any custom experiments.
The `run.py` script will fail if the processed dataset and curvature SDRs are not found. If it is not found, then follow the script's instructions
to generate this dataset (with the help of scripts like `process_data.py`, `curvature_encoder.py`, `generate_paths.py` and/or `generate_random.py`).

## Tutorial

### Data generation

Test to see if the conda installation is successful by checking to see if the following `import` works:

```python```

```from nupic.bindings.math import SparseMatrixConnections```

Ensure that trimesh is the correct version: `pip install trimesh==3.10.8`. Later versions of trimesh don't always produce the correct results for these experiments.

#### Flags

`sdr_p`: where to save the dataset

`ycb_p`: root directory of the YCB path (most likely `~/tbp/data/habitat/objects/ycb`)

`-r`: hash radius in Cartesian Coordinate space (ultimately determines how much one SDR overlaps with another)

`-objects`: list of objects (separated by a space) to save data for

`-n`: the total number of bits in the SDR in the case of the `curvature_encoder.py`

`-w`: total number of *ON* bits in the SDR in the case of the `curvature_encoder.py`

`-num_paths`: number of paths to enerate for training and testing in the case of `generate_paths.py`

`-path_size`: length of each path to generate for training and test in the case of `generate_paths.py`

`-num_points`: number of uniformly random points to generate for training and testing in the case of `generate_random.py`

#### Pre-process the data

Update `<YOU>` in the following lines to reflect your user ID. Replace other arguments with `< >` and specify your own. The following are single-line commands.

```python ~/tbp/tbp.monty/projects/grid_cells/process_data.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/grid_cells/grid_dataset -ycb_p ~/tbp/data/habitat/objects/ycb```

```python ~/tbp/tbp.monty/projects/grid_cells/curvature_encoder.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/grid_cells/grid_dataset -r 5 -objects <0 1 2 3...> -n 1024 -w 11```

Run either `generate_paths.py` to train on sequences of somewhat continuous paths along an object's surface. Or run `generate_random.py` to train on uniformly random points along an object's surface.

```python ~/tbp/tbp.monty/projects/grid_cells/generate_paths.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/grid_cells/grid_dataset -objects <0 1 2 3...> -num_paths 50 -path_size 10```

```python ~/tbp/tbp.monty/projects/grid_cells/generate_random.py -sdr_p /Users/<YOU>/tbp/tbp.monty/projects/grid_cells/grid_dataset -objects <0 1 2 3...> -num_points 500```

### Run the experiment

```python tbp.monty/projects/monty_runs/run.py -e <CONFIG NAME>```. For example, you can run ```python run.py -e htm_random_path_base``` if you ran `generate_paths.py` or ```python run.py -e htm_random_points_base``` if you ran `generate_random.py`.