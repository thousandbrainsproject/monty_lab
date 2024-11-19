# General
This folder contains run files and experiment configs for running embodied Monty experiments (`run.py`) and parallel experiments (`run_parallel.py`). 

To run an experiment run `python experiments/run.py -e my_experiment` (or `run_parallel.py` to use multiprocessing).

## Graph Experiments

There are three types of custom Learning Modules in the implementation:

* `DisplacementGraphLM`: Uses displacemens to recognize objects and their poses.
* `FeatureGraphLM`: Uses features at locations to recognize objects and their poses.
* `EvidenceGraphLM`: Has real valued evidence count that is updated for each pose on an object.
  
A detailed description of this code and the custom classes can be found [here](https://drive.google.com/file/d/16MMb0BUIQvEKX6YC9XIwEOVzqtjkvfnB/view?usp=sharing). (See [Overleaf](https://www.overleaf.com/read/qxchttxzpfnd) for most recent version.)

### Logging

Experiment results can be logged to a .csv file, a more detailed .json file, and to weights and biases. The level and type of logging can be specified by adding logging handlers to the logging_config of an experiment.

NOTE: When logging detailed stats to wandb the logged animations do not appear in the dashboard. Currently they can be found in the runs artifacts folder.

To analyze results use functions from `plot_utils.py`. For examples of how to use them see the jupyter notebooks in `projects/graphs/` (good starting points are `AnalyzeResults.ipynb`, `EvalAnalysis.ipynb`, and `EvidenceLM.ipynb`).

## Follow-up Configs
If you are trying to debug something or simply want to learn more about what is happening during an experiment you can use the `make_detailed_follow_up_configs.py` script. This script will generate a config for rerunning one or several episodes of a previous experiment with detailed logging. You can then visualize and analyze the detailed logs. We do not recommend running an entire benchmark experiment with detailed logging since the log files will become prohibitively large.

### More Information
For more information on running experiments, logging, analysis and the kind of modules we have implemented, please refer to our extensive documentation: https://thousandbrainsproject.readme.io/docs/getting-started 