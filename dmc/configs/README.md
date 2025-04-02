# Experiment Configs for Key Figures

Below is a summary of configs that correspond to figures in the Demonstrating Monty Capabilities paper, with descriptions motivating the choice of config parameters.

## Figure 1 & 2: Diagramatic Figures With No Experiments

## Figure 3: Robust Sensorimotor Inference

This figure presents results from four inference experiments testing Monty's robustness under different conditions. Monty was pre-trained on 14 standard rotations derived from cube face and corner views (see full configuration details in `pretraining_experiments/dist_agent_1lm`).

- `dist_agent_1lm`: Standard inference with no sensor noise or random rotations
- `dist_agent_1lm_noise`: Tests robustness to sensor noise
- `dist_agent_1lm_randrot_all`: Tests performance across 14 random rotations
- `dist_agent_1lm_randrot_all_noise`: Tests performance with both random rotations and sensor noise

Here we are showing the performance of the "standard" version of Monty, using:
- 77 objects
- 14 rotations
- Goal-state-driven/hypothesis-testing policy active
- A single LM (no voting)

The main output measures are accuracy, rotation error (degrees), and Chamfer distance for each condition.

Figure 3 also include an analysis of symmetric rotations, the data for which is generated with the experiment `fig3_symmetry_run` which is defined in `visualizations.py`.

## Default Parameters for Figures 4+
Unless specified otherwise, the following figures/experiments use:
- 77 objects
- 5 predefined "random" rotations. These rotations were randomly generated but are kept constant across experiments. 
- Sensor noise

This captures core model performance in a realistic setting.

## Figure 4: Structured Object Representations

Consists of 1 experiment:
- `surf_agent_1lm_randrot_noise_10simobj`

This means performance is evaluated with:
- 10 morphologically similar objects
- 5 random rotations
- Sensor noise
- Hypothesis-testing policy active
- No voting

The main output measure is a dendrogram showing evidence score clustering for the 10 objects.

**Notes:**
- Although evaluating on 10 objects, the model is trained on 77 objects.
- We need to run this experiment with SELECTIVE logging on so we get the evidence values to analyze.
  
## Figure 5: Rapid Inference with Voting

Consists of 9 experiments:
- `dist_agent_1lm_randrot_noise`
- `dist_agent_2lm_half_lms_match_randrot_noise`
- `dist_agent_4lm_half_lms_match_randrot_noise`
- `dist_agent_8lm_half_lms_match_randrot_noise`
- `dist_agent_16lm_half_lms_match_randrot_noise`
- `dist_agent_2lm_fixed_min_lms_match_randrot_noise`
- `dist_agent_4lm_fixed_min_lms_match_randrot_noise`
- `dist_agent_8lm_fixed_min_lms_match_randrot_noise`
- `dist_agent_16lm_fixed_min_lms_match_randrot_noise`

There are two variations, either
- `half_lms_match`: At least half of the available LMs must match. This tends to improve accuracy but results in a slower increase in convergence speed as a function of matching steps.
  - Defined as: `min_lms_match=int(num_lms/2)`
- `fixed_min_lms_match`: A minimum of two LMs must match, regardless of the total number of LMs. This tends to increase convergence speed significantly but provides only a small improvement in accuracy.
  - Defined as: `min_lms_match=min(num_lms, 2)`

Performance is evaluated on:
- 77 objects
- Goal-state-driven/hypothesis-testing policy active
- Sensor noise and 5 random rotations
- Voting over 2, 4, 8, or 16 LMs

The main output measure is accuracy and rotation error as a function of number of LMs. The two variations show that accuracy and convergence speed can be traded off against each other.

## Figure 6: Rapid Inference with Model-Free and Model-Based Policies

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

## Figure 7: Rapid Learning

Consists of 7 experiments:
- `pretrain_dist_agent_1lm_checkpoints`
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

## Figure 8: Computationally Efficient Learning and Inference

### Inference (12 experiments)

There are two sets of experiments, one using hypothesis testing and another using no hypothesis testing.

- `dist_agent_1lm_randrot_nohyp_x_percent_5p` - 5% threshold (No Hypothesis Testing)
- `dist_agent_1lm_randrot_nohyp_x_percent_10p` - 10% threshold (No Hypothesis Testing)
- `dist_agent_1lm_randrot_nohyp_x_percent_20p` - 20% threshold (No Hypothesis Testing)
- `dist_agent_1lm_randrot_nohyp_x_percent_40p` - 40% threshold (No Hypothesis Testing)
- `dist_agent_1lm_randrot_nohyp_x_percent_60p` - 60% threshold (No Hypothesis Testing)
- `dist_agent_1lm_randrot_nohyp_x_percent_80p` - 80% threshold (No Hypothesis Testing)
- `dist_agent_1lm_randrot_x_percent_5p` - 5% threshold (With Hypothesis Testing)
- `dist_agent_1lm_randrot_x_percent_10p` - 10% threshold (With Hypothesis Testing)
- `dist_agent_1lm_randrot_x_percent_20p` - 20% threshold (With Hypothesis Testing)
- `dist_agent_1lm_randrot_x_percent_40p` - 40% threshold (With Hypothesis Testing)
- `dist_agent_1lm_randrot_x_percent_60p` - 60% threshold (With Hypothesis Testing)
- `dist_agent_1lm_randrot_x_percent_80p` - 80% threshold (With Hypothesis Testing)

**Notes:**

- For the 12 experiments above, `x_percent_threshold` determines the threshold at which the LM determines it has converged. In addition, we explicitly set `evidence_update_threshold="80%".

This performance is evaluated with:

- 77 objects
- 5 random rotations
- No sensor noise*
- No voting

*Due to ViT model comparison.

The main output measure is accuracy and FLOPs as a function of `x_percent threshold` and whether hypothesis testing was used or not.

## Pretraining Experiments
`pretraining_experiments.py` defined pretraining experiments that generate models
used throughout this repository. They are required for running eval experiments,
visualization experiments, and for many of the figures generated in the `scripts`
directory. The following is a list of pretraining experiments and the models they produce:
 - `pretrain_dist_agent_1lm` -> `dist_agent_1lm`
 - `pretrain_surf_agent_1lm` -> `surf_agent_1lm`
 - `pretrain_dist_agent_2lm` -> `dist_agent_2lm`
 - `pretrain_dist_agent_4lm` -> `dist_agent_4lm`
 - `pretrain_dist_agent_8lm` -> `dist_agent_8lm`
 - `pretrain_dist_agent_16lm` -> `dist_agent_16lm`

All of these models are trained on 77 YCB objects with 14 rotations each (cube face
and corners).

## Visualization Experiments

`visualizations.py` contains configs defined solely for making visualizations that go into
paper figures. The configs defined are:
- `fig2_pretrain_surf_agent_1lm_checkpoints`: A pretraining experiment that saves
  checkpoints for the 14 training rotations. The output is read and plotted by
  functions in `scripts/fig2.py`.
- `fig3_evidence_run`: A one-episode distant agent experiment used to collect evidence
   and sensor data for every step. The output is read and plotted by functions in
    `scripts/fig3.py`.
- `fig4_symmetry_run`: Runs `dist_agent_1lm_randrot_noise` with storage of
   evidence and symmetry including symmetry data for the MLH object only, and only
   for the terminal step of each episode. The output is read and plotted by
   functions in `scripts/fig4.py`.
- `fig5_visualize_8lm_patches`: A one-episode, one-step experiment that is used to
  collect one set of observations for the 8-LM model. The output is read and plotted
  by functions in `scripts/fig5.py` to show how the sensors patches fall on the object.
- `fig6_curvature_guided_policy`: A one-episode surface agent experiment with
  no hypothesis-testing policy active. The output is read and plotted by
  functions in `scripts/fig6.py`.
- `fig6_hypothesis_driven_policy`: A one-episode surface agent experiment with
  hypothesis-testing policy active. The output is read and plotted by
  functions in `scripts/fig6.py`.

All of these experiments should be run in serial due to the memory needs of
detailed logging (or checkpoint-saving in the case of
`fig2_pretrain_surf_agent_1lm_checkpoints`).

All experiments save their results to subdirectories of `DMC_ROOT` / `visualizations`.

## Other Experiments
`view_finder_experiments.py` defines five experiments:
- view_finder_base: 14 standard training rotations
- view_finder_randrot_all: 14 randomly generated rotations
- view_finder_randrot: 5 pre-defined "random" rotations
- view_finder_32: 32 training rotations for rapid learning experiments
- view_finder_base_highres: 14 standard training rotations at 512x512 resolution.
  
Note these experiments are not used for object recognition in Monty. Rather, they use Monty to capture and store images of objects in the YCB dataset. The first four experiments are used for generating training/testing data for the ViT-based model, and the last experiment (`view_finder_base_highres`) is used to capture higher resolution images used in some figures and illustrations. Arrays stored during these experiments can be rendered by `scripts/render_view_finder_images.py`.