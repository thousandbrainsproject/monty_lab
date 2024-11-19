# Surface modeling and pose optimization

This project contains prototyping code assocaited with three problems: learning a surface model of an object, learning to align a set of observations (e.g. point cloud) to a template (e.g. a point cloud in some canonical pose), and using the former two to aid the overarching problem of inferring object id and pose simultaneously.

# Env

This project folder may use packages during development that have not been added as tbp.monty dependencies. To create an environment for this folder, please run 

`conda create --clone tbp.monty -n dev_3d`

From here, consider installing pytorch3dsimply with

`pip install pytorch3d`.

# Experiments

## Iterative closest point

Iterative closest point (ICP) is an algorithm for estimating the pose of two point clouds to one another by trying to align them so they are indistinguishable. To run ICP on modelnet40 objects with various rotations applied, you can run either 

`python projects/surface_modeling/run.py -e icp_pose_optimization_single_axis_rotation`

and replace the field after -e with whatever experiment you want from experiments.py. To run a bunch of experiments in a row, use run_exps.sh.

To visualize the results, use
`python projects/surface_modeling/analyze_exps.py`
You will need to manually type in the path to the experiment you want to analyze in this file.

Note that MCMC experiments are intended as a baseline, and due to rapid changes, are unstable and likely broken.

## Poisson surface reconstruction

Poisson surface reconstruction converts a point cloud into an implicit surface. To run poisson surface reconstruction on modelnet40 objects, run

`python ./poisson_modelnet_40.py`

To edit the number of training or testing points, edit the code under the main conditional. This file also contains functions to visualize reconstructed surfaces, and also to loop over parameters like depth and num_points.
