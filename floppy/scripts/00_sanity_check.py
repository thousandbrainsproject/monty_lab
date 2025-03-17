import os
from pathlib import Path

import pandas as pd

experiments = [
    "dist_agent_1lm_randrot_nohyp_x_percent_5p",
    "dist_agent_1lm_randrot_nohyp_x_percent_10p",
    "dist_agent_1lm_randrot_nohyp_x_percent_20p",
    "dist_agent_1lm_randrot_nohyp_x_percent_30p_threaded",
    # "dist_agent_1lm_randrot_nohyp_x_percent_30p_evidence_update_all",
]


def number_of_flop_traces_files(experiment_dir: Path) -> int:
    files = os.listdir(experiment_dir)
    return len([file for file in files if file.startswith("flop_traces")])


experiment_dfs = {}
for experiment in experiments:
    experiment_dir = (
        Path(f"~/tbp/results/dmc/results/{experiment}").expanduser().resolve()
    )
    file_counter = number_of_flop_traces_files(experiment_dir)
    print(f"Experiment {experiment} has {file_counter} flop_traces files")
    dfs = []
    for file in os.listdir(experiment_dir):
        if file.startswith("flop_traces"):
            df = pd.read_csv(experiment_dir / file)
            dfs.append(df)
    combined_df = pd.concat(dfs)
    experiment_dfs[experiment] = combined_df

# For each experiment_dfs, print len(experiment_dfs[experiment])
for experiment in experiments:
    # print(f"Experiment {experiment} has {len(experiment_dfs[experiment])} total rows")
    # Print number of rows where method == "experiment.run_episode"
    run_episode_df = experiment_dfs[experiment][
        experiment_dfs[experiment]["method"] == "experiment.run_episode"
    ]
    run_epoch_df = experiment_dfs[experiment][
        experiment_dfs[experiment]["method"] == "experiment.run_epoch"
    ]
    # Sum the flops of the run_episode_df
    run_episode_flops = run_episode_df["flops"].sum()
    run_epoch_flops = run_epoch_df["flops"].sum()

    # Number of episodes
    num_episodes = run_episode_df["episode"].nunique()
    num_epochs = run_epoch_df["episode"].nunique()

    exp_name = experiment.split("_")[-1]
    print(
        f"Experiment {exp_name} has TOTAL {num_episodes} EPISODE FLOPS: {run_episode_flops / 1e9:.2f} B flops"
    )
    print(
        f"Experiment {exp_name} has TOTAL {num_epochs} EPOCH FLOPS: {run_epoch_flops / 1e9:.2f} B flops"
    )

# file_counter = 0
# for file in files:
#     if file.startswith("flop_traces"):
#         df = pd.read_csv(experiment_dir / file)
#         combined_df = pd.concat([combined_df, df])
#         file_counter += 1

# print(f"Read {file_counter} flop_traces files")

# # Extract experiment.run_episode flops
# run_episode_df = combined_df[combined_df["method"] == "experiment.run_episode"]
# # Set the flops to human readable
# run_episode_df["flops_human_readable"] = run_episode_df["flops"] / 1e9
# # Sort by episode and save to csv
# run_episode_df = run_episode_df.sort_values(by="episode")
# # Drop timestamp, lineno, filename, parent_method
# run_episode_df = run_episode_df.drop(
#     columns=["timestamp", "line_no", "filename", "parent_method"]
# )
# run_episode_df.to_csv(experiment_dir / "run_episode_flops.csv", index=False)

# # Unique values in method column
# unique_methods = combined_df["method"].value_counts()
# print(unique_methods)

# # Unique episode ids
# unique_episode_ids = combined_df["episode"].value_counts()
# print(unique_episode_ids)

# # Select episode == 1
# episode_1_df = combined_df[combined_df["episode"] == 1]
# # Sum the flops where method == "monty._matching_step"
# flops_sum = episode_1_df[episode_1_df["method"] == "monty._matching_step"][
#     "flops"
# ].sum()

# matching_step_flops = flops_sum / 1e9
# # Print it human readable
# print(f"Matching Step:{matching_step_flops:.2f}B")

# # Print flops for method = experiment.pre_episode
# pre_episode_df = episode_1_df[episode_1_df["method"] == "experiment.pre_episode"]
# pre_episode_flops = pre_episode_df["flops"].sum() / 1e9
# print(f"Pre Episode:{pre_episode_flops:.2f}B")

# # Print flops for method = experiment.post_episode
# post_episode_df = episode_1_df[episode_1_df["method"] == "experiment.post_episode"]
# post_episode_flops = post_episode_df["flops"].sum() / 1e9
# print(f"Post Episode:{post_episode_flops:.2f}B")

# # Sum the flops for the matching step, pre episode, and post episode
# total_flops = matching_step_flops + pre_episode_flops + post_episode_flops
# print(f"Total Flops:{total_flops:.2f}B")

# # Print the flops for method = experiment.run_episode
# run_episode_df = episode_1_df[episode_1_df["method"] == "experiment.run_episode"]
# run_episode_flops = run_episode_df["flops"].sum() / 1e9
# print(f"Run Episode:{run_episode_flops:.2f}B")

# # Compare total flops to the run_episode flops
# print(f"Total Flops / Run Episode Flops: {total_flops / run_episode_flops:.2f}")

# # In the combined_df, how many monty._matching_step flops are more than 0 for each episode
# result = {}
# for episode in sorted(combined_df["episode"].unique()):
#     matching_step_df = combined_df[
#         (combined_df["method"] == "monty._matching_step")
#         & (combined_df["episode"] == episode)
#     ]
#     # Count number of rows in matching_step_df that have flops > 0
#     matching_step_df = matching_step_df[matching_step_df["flops"] > 0]
#     print(f"Episode {episode} has {len(matching_step_df)} matching step flops")
#     result[episode] = len(matching_step_df)

# # Save the result to a csv
# pd.DataFrame(result.items(), columns=["episode", "matching_step_flops"]).to_csv(
#     f"{experiment_dir}/matching_step_flops.csv", index=False
# )

# Read the run_episode_flops.csv file for the two experiments
# experiment_1 = "dist_agent_1lm_randrot_nohyp_x_percent_30p"
# experiment_2 = "dist_agent_1lm_randrot_nohyp_x_percent_5p"
# run_episode_flops_1 = pd.read_csv(
#     Path(f"~/tbp/results/dmc/results/{experiment_1}/run_episode_flops.csv")
#     .expanduser()
#     .resolve()
# )
# run_episode_flops_2 = pd.read_csv(
#     Path(f"~/tbp/results/dmc/results/{experiment_2}/run_episode_flops.csv")
#     .expanduser()
#     .resolve()
# )

# # Sum the flops for each experiment
# print(run_episode_flops_1["flops_human_readable"].sum())
# print(run_episode_flops_2["flops_human_readable"].sum())

# # What about the mean?
# print(run_episode_flops_1["flops_human_readable"].mean())
# print(run_episode_flops_2["flops_human_readable"].mean())

# # What about the median?
# print(run_episode_flops_1["flops_human_readable"].median())
# print(run_episode_flops_2["flops_human_readable"].median())

# # Do the same for flops columns dont't use human readable
# print(run_episode_flops_1["flops"].sum())
# print(run_episode_flops_2["flops"].sum())

# # What about the mean?
# print(run_episode_flops_1["flops"].mean())
# print(run_episode_flops_2["flops"].mean())

# # What about the median?
# print(run_episode_flops_1["flops"].median())
# print(run_episode_flops_2["flops"].median())
