"""WIP script to extract partial or full graph object models.

The ultimate goal of the script is to extract points observed by Monty at
various steps (or at least when it terminates) during evaluation experiments.

We also want to quantitatively calculate the proportion of surface area uncovered
by Monty during evaluation experiments.

For Phase I, I will:
1. Extract points for full object models from pretrained models.
2. Write a function to calculate the surface area of the object models.

Phase I will help me identify exactly what will need to be extracted from
evaluation experiments, as well have the surface area calculation function for
full or partial object models.

For Phase II, I will:
1. Need to modify `tbp.monty` to save point clouds at evaluation steps.
"""
