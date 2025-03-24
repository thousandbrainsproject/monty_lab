# Welcome to the Monty Lab Repository!

This is where we (the TBP team) keep our day-to-day experiment files and data analysis scripts. The code in this repository is not unit tested or actively maintained. Some older projects in here are just for informational purposes or to replicate old experiments.

## Setup

To run the code in this repository requires Conda to install the dependencies. For instructions on how to install Conda (Miniconda or Anaconda) on your machine see https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

NOTE: By default, Conda will activate the base environment when you open a new terminal. If you do not want Conda to change your global shell when you open a new terminal, run: `conda config --set auto_activate_base false`.

To setup `monty_lab`, **use the conda commands below**. Make sure to `cd` into the `monty_lab` directory before running these commands.

Note that the commands are slightly different depending on whether you are setting up the environment on an Intel or ARM64 architecture, and whether you are using the zsh shell or another shell.

You can create the environment with the following commands:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init zsh
conda activate monty_lab
conda config --env --set subdir osx-64
```

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init
conda activate monty_lab
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate monty_lab
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate monty_lab
```

Then, you can install any additional dependencies your specific project requires like this:

`pip install jupyter`

If your project requires several additional dependencies you should document this in the project's README or as a requirement.txt file in that folder.

## Content
- [**experiments**](./experiments): A collection of configs for different experiments, including the basic run.py and run_parallel.py scripts.
- [**graph_matching**](./graph_matching): A collection of notebooks for analyzing graph matching experiments.
- [**grid_cells**](./grid_cells): A project investigating the use of grid cells to encode 3D locations and use these to recognize 3D objects.
- [**high_dim_coincidence_detection**](./high_dim_coincidence_detection): A project investingating the use of HDCD to detect objects and their poses.
- [**monty_meets_world**](./monty_meets_world): An iOS app to collect data for Monty (offline or streaming) from an iPad or iPhone camera. Includes a notebook to visualize this data.
- [**object_behaviors**](./object_behaviors): An environment based on sympy to simulate simple moving objects. We plan to use this to test modeling object behaviors in Monty.
- [**speedup**](./speedup): A collection of notebooks investigating potential speedups in the Monty algorithm.
- [**surface_modeling**](./surface_modeling): An early project attempting an alternative way of recognizing objects and poses using ICP. Note that this approach required collecting a batch of data which is not in the vein of Monty's sensorimotor approach. It can be used as a comparison though.
- [**synthetic_sdr**](./synthetic_sdr): A project implementing SDR generation from similarity matrices. This was the basis for the EvidenceSDRLMMixing, integrated in the Monty framework code.
- [**tactile_temporal_memory**](./tactile_temporal_memory): A project testing the temporal memory algorithm on data collected with the surface agent in habitat.
- [**temporal_memory**](./temporal_memory): A first implementation of the HTM algorithm for recognizing 3D objects. Includes data preprocessing and SDR generation for 3D locations. Note: This code was written when the Monty framework was not well established yet.
- [**touch_sensor**](./touch_sensor): The first implementation of a sensor moving along the surface of an object. This is now integrated in the Monty framework and called "surface agent".
- [**very_first_experiments**](./very_first_experiments): A couple of notebooks visualizing first experiments of working with sensorimotor data collected in habitat and building and extending graphs using this data.
- [**dmc_configs**](./dmc_configs): A collection of configs for experiments found in the "Demonstrating Monty's Capabilities" paper.

## Contributing
This repository is not meant for external contributors. It is just a place where the TBP team keeps its project files. If you are on the TBP team and start working on a new project, please remember to always add a README in your project folder and a short description in the "Content" section here.

If you commit code to this repository, please make sure it is properly formatted. We currently don't have automated checks for this in place. Since this project currently doesn't have a `pyproject.toml` file, to check or fix formatting you will need to run `ruff` with the `--config` option, for example: `ruff check --config ../tbp.monty/pyproject.toml .`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
