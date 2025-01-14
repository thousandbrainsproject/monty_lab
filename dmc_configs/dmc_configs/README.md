# Experiment Configs for Key Figures

Below is a summary of configs that correspond to figures in the Demonstrating Monty Capabilities paper, with descriptions motivating the choice of config parameters.

**Note:** In all experiments, `use_semantic_sensor=False` should be specified. This should be the case once PR #116 is merged, updating the values for PatchViewFinderMountHabitatDatasetArgs etc to be False by default.

## Figure 1 & 2: Diagramatic Figures With No Experiments

## Figure 3: Robust Sensorimotor Inference

Consists of 4 experiments:
- `dist_agent_1lm` (i.e. no noise)
- `dist_agent_1lm_noise` - Sensor noise
- `dist_agent_1lm_randrot` - 5 random rotations (5 rather than e.g. 14 or 32 due to the long time to run the experiments)
- `dist_agent_1lm_randrot_noise`

Here we are showing the performance of the "standard" version of Monty, using:
- 77 objects
- Goal-state-driven/hypothesis-testing policy active
- A single LM (no voting)

The main output measure is accuracy and rotation error as a function of noise conditions.

## Default Parameters for Figures 4+
Unless specified otherwise, the following figures/experiments use:
- 77 objects
- 5 random rotations
- Sensor noise

This captures core model performance in a realistic setting.

## Figure 4: Rapid Inference with Voting

Consists of 5 experiments:
- `dist_agent_1lm_randrot_noise`
- `dist_agent_2lm_randrot_noise`
- `dist_agent_4lm_randrot_noise`
- `dist_agent_8lm_randrot_noise`
- `dist_agent_16lm_randrot_noise`

This means performance is evaluated with:
- 77 objects
- Goal-state-driven/hypothesis-testing policy active
- Sensor noise and 5 random rotations
- Voting over 1, 2, 4, 8, or 16 LMs

The main output measure is accuracy and rotation error as a function of number of LMs.

**TODO:**
- Config builders for arbitrary numbers of LMs are not currently included in `dmc_eval_experiments.py`.
- Comparable configs should be used to generate views for evaluated rotations to pass to ViT model for comparison.

## Figure 5: Rapid Inference with Model-Based Policies

Consists of 3 experiments:
- `dist_agent_1lm_randrot_noise_nohyp` - No hypothesis-testing
- `dist_agent_1lm_randrot_noise_moderatehyp` - Occasional hypothesis-testing
  - Should specify:
    - elapsed_steps_factor=20
    - min_post_goal_success_steps=10
- `dist_agent_1lm_randrot_noise` - Default, frequent hypothesis-testing
  - Should specify:
    - elapsed_steps_factor=10
    - min_post_goal_success_steps=5

This means performance is evaluated with:
- 77 objects
- Sensor noise and 5 random rotations
- No voting
- Varying levels of hypothesis-testing

The main output measure is accuracy and rotation error as a function of hypothesis-testing policy.

**TODO:**
- These configs need to be specified.

## Figure 6: Rapid Learning

Consists of 6 experiments:
- `dist_agent_1lm_randrot_nohyp_1rot_trained`
- `dist_agent_1lm_randrot_nohyp_2rot_trained`
- `dist_agent_1lm_randrot_nohyp_4rot_trained`
- `dist_agent_1lm_randrot_nohyp_8rot_trained`
- `dist_agent_1lm_randrot_nohyp_16rot_trained`
- `dist_agent_1lm_randrot_nohyp_32rot_trained`

This means performance is evaluated with:
- 77 objects
- 5 random rotations
- NO sensor noise*
- NO hypothesis-testing*
- No voting
- Varying numbers of rotations trained on (evaluations use different baseline models)

*No hypothesis-testing as the ViT model comparison only receives one view and cannot move around object, and no noise since Sensor-Module noise does not have a clear analogue for the ViT model.

The main output measure is accuracy and rotation error as a function of training rotations.

**Notes:**
- Training rotations should be structured:
  1. First 6 rotations = cube faces
  2. Next 8 rotations = cube corners
  3. Remaining = random rotations (as otherwise introduces redundancy)

**TODO:**
- Configs need to be specified
- Comparable configs needed for ViT model comparison

## Figure 7: Computationally Efficient Learning and Inference

Consists of 8 experiments:

### Inference (7 experiments):
- `dist_agent_1lm_randrot_nohyp_x_percent_5p` - 5% threshold
- `dist_agent_1lm_randrot_nohyp_x_percent_10p` - 10% threshold
- `dist_agent_1lm_randrot_nohyp_x_percent_20p` - 20% threshold (default for other experiments)
- `dist_agent_1lm_randrot_nohyp_x_percent_40p` - 40% threshold
- `dist_agent_1lm_randrot_nohyp_x_percent_80p` - 80% threshold
- `dist_agent_1lm_randrot_nohyp_x_percent_100p` - 100% threshold

This means performance is evaluated with:
- 77 objects
- 5 random rotations
- No sensor noise*
- No hypothesis-testing*
- No voting

*Due to ViT model comparison.

The main output measure is accuracy and FLOPs as a function of x-percent threshold.

### Training (1 experiment):
- `dist_agent_77obj_1rot_trained`**

**Single rotation evaluation due to FLOPs counting overhead, so we will extrapolate total FLOPs to 14 rotations based on this/or can compare to a ViT trained on 1 rotation.

The main output measure is as a function of whether the ViT or Monty is training.

**TODO:**
- Configs need specification (including `dist_agent_77obj_1rot_trained` in `dmc_pretrain_experiments.py`)
- Comparable config needed to generate views corresponding to the 5 random, evaluated rotations, which can then be passed to the ViT model(s) for comparison.

## Figure 8: Multi-Modal Transfer

Consists of 4 experiments:
- `dist_agent_1lm_randrot_noise` - "Standard" Monty ("dist_on_dist")
- `dist_on_touch_1lm_randrot_noise` - "dist_on_touch"
- `touch_agent_1lm_randrot_noise` - "touch_on_touch" baseline
- `touch_on_dist_1lm_randrot_noise` - "touch_on_dist"

This means performance is evaluated with:
- 77 objects
- 5 random rotations
- Sensor noise
- Hypothesis-testing policy active
- No voting

The main output measure is accuracy and rotation error for within/across modality inference.

## Figure 9: Structured Object Representations

Consists of 1 experiment:
- `dist_agent_1lm_randrot_noise_10simobj`

This means performance is evaluated with:
- 10 morphologically similar objects
- Random rotations
- Sensor noise
- Hypothesis-testing policy active
- No voting

The main output measure is a dendrogram showing evidence score clustering for the 10 objects.

**Notes:**
- Although evaluating on 10 objects, the model is trained on 77 objects.
- We need to run this experiment with SELECTIVE logging on so we get the evidence values to analyze.

**TODO:**
- Config needs specification, including training config for similar objects in `dmc_pretrain_experiments.py`.