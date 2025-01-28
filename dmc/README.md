# Demonstrating Monty Capabilities Experiments

This repository contains the code for the experiments in the paper "Demonstrating Monty Capabilities".

## Setup

The results of DMC experiments are stored under `DMC_ROOT_DIR`. By default, this is `~/tbp/results/dmc`, but you can change this by setting the `DMC_ROOT_DIR` environment variable. It has the following structure:

```
DMC_ROOT_DIR/
    pretrained_models/
    results/
    view_finder_images/        
```

The `pretrained_models` directory contains the pre-trained models for Monty experiments. The `results` directory contains the results of evaluation experiments. The `view_finder_images` directory contains the images used for the view finder experiments.

Figures and tables are stored under `DMC_ANALYSIS_DIR`. By default, this is `~/tbp/results/dmc_analysis`, but you can change this by setting the `DMC_ANALYSIS_DIR` environment variable.

## Running the Experiments

Configs to train models and run experiments are in the `configs` directory.

## Reproducting Figures from the Paper

Analaysis scripts to generate the figures are in the `scripts` directory. The data for figures can either be generated via `configs`, or you can directly download the pre-trained models at `link-to-be-confirmed` and the raw results from experiments at `link-to-be-confirmed`.
