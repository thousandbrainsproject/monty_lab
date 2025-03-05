from pathlib import Path

import pandas as pd
from configs import CONFIGS  # noqa: E402
from configs.common import DMC_PRETRAIN_DIR
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.run import main  # noqa: E402
from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict

setup_env()


def summarize(exp_dir: Path, title: str):
    exp_dir = Path(exp_dir)
    df = pd.read_csv(exp_dir / "eval_stats.csv")
    n_episodes = len(df)
    vc = dict(df["primary_performance"].value_counts())
    print(title)
    print(f" - Correct: {100 * vc['correct'] / n_episodes}")
    print(f" - Correct MLH: {100 * vc['correct_mlh'] / n_episodes}")
    print(f" - Correct*: {100 * (vc['correct'] + vc['correct_mlh']) / n_episodes}")
    with open(exp_dir / "parallel_log.txt", "r") as f:
        ln = f.readlines()[-1]
        total_secs = float(ln.split(": ")[-1])
    mins, secs = divmod(total_secs, 60)
    print(f" - Total time: {mins:.0f}m {secs:.0f}s")
    sec_per_episode = total_secs / n_episodes
    print(f" - Time per episode: {sec_per_episode:.2f}s\n")


path = Path.home() / "tbp/results/dmc/results.80_80/dist_agent_1lm"
summarize(path, "x_percent_threshold=80, nneighbors=10")

path = Path.home() / "tbp/results/dmc/results.80_50/dist_agent_1lm"
summarize(path, "x_percent_threshold=50, nneighbors=10")

path = Path.home() / "tbp/results/dmc/results.80_20/dist_agent_1lm"
summarize(path, "x_percent_threshold=20, nneighbors=5")


# experiment_name = "fig4_visualize_8lm_patches"

# config = CONFIGS[experiment_name]
# config_dict = config_to_dict(config)
# exp = config["experiment_class"]()
# exp.setup_experiment(config_dict)
# h = exp.logger_handler.loggers[0].handlers[1]
pass

# config["experiment_args"].n_eval_epochs = 1
# config["experiment_args"].model_name_or_path = str(
#     DMC_PRETRAIN_DIR / "dist_agent_1lm_10distinctobj/pretrained"
# )
# config["monty_config"].learning_module_configs["learning_module_0"][
#     "learning_module_args"
# ]["use_multithreading"] = False
# config.update(
#     dict(
#         eval_dataloader_class=ED.InformedEnvironmentDataLoader,
#         eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
#             object_names=["mug"],
#             object_init_sampler=PredefinedObjectInitializer(
#                 rotations=[[196, 326, 225]]
#             ),
#         ),
#     )
# )
# main(all_configs=CONFIGS, experiments=[experiment_name])
