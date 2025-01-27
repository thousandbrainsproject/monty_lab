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

With two variations, either
- Half the number of LMs must match; this will tend to increase accuracy, but with a smaller improvement in convergence as a function of matching steps.
  - `min_lms_match=int(num_lms/2)`
- Or, the minimum number of LMs that must match is 2; this will tend to increase convergence very quickly, but with a smaller/minimal improvement in accuracy.
  - `min_lms_match=min(num_lms, 2)`

This means performance is evaluated with:
- 77 objects
- Goal-state-driven/hypothesis-testing policy active
- Sensor noise and 5 random rotations
- Voting over 1, 2, 4, 8, or 16 LMs

The main output measure is accuracy and rotation error as a function of number of LMs. The two variations show that accuracy and convergence speed can be traded off against each other.

## Figure 5: Rapid Inference with Model-Free and Model-Based Policies

Consists of 3 experiments:
- `dist_agent_1lm_randrot_noise_nohyp` - No hypothesis-testing, and random-walk policy
- `surf_agent_1lm_randrot_noise_nohyp` - Model-free policy to explore surface
- `surf_agent_1lm_randrot_noise` - Default, i.e. model-free and model-based policies

This means performance is evaluated with:
- 77 objects
- Sensor noise and 5 random rotations
- No voting
- Varying policies; the surface agent (i.e. with color etc) gets the same kind of sensory information as the distant agent, and so differs only in its model-free policy that encourages rapid exploration of the surface of the object. We can make it clear in the paper that there is nothing preventing the distant agent from also having model-free and model-based policies.

The main output measure is accuracy and rotation error as a function of the policy used.

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

## Figure 7: Computationally Efficient Learning and Inference

Consists of 8 experiments:

### Inference (7 experiments):
- `dist_agent_1lm_randrot_nohyp_x_percent_5p` - 5% threshold
  - `evidence_update_threshold=x_percent_threshold`  # Evidence update threshold is the same as x-percent threshold for determining convergence.
- `dist_agent_1lm_randrot_nohyp_x_percent_10p` - 10% threshold
- `dist_agent_1lm_randrot_nohyp_x_percent_15p` - 15% threshold
- `dist_agent_1lm_randrot_nohyp_x_percent_20p` - 20% threshold (default for other experiments)
- `dist_agent_1lm_randrot_nohyp_x_percent_30p` - 30% threshold
- `dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all`
  - 30% x-percent threshold for *convergence*
  - Effectively >100% threshold for which hypotheses to test, i.e. all hypotheses are tested, even those with negative evidence; determined by setting `evidence_update_threshold='all'
  
**Notes:**
- For the first 5 experiments above, x-percent threshold determines the threshold at which the LM determines it has converged. In addition we set `evidence_update_threshold=x_percent_threshold`, so that this same threshold also determines which evidence values are updated. 
- For the final experiment, these are separated out as noted.

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

The main output measure is FLOPs as a function of whether the ViT or Monty is training.

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