from tbp.monty.frameworks.run_env import setup_env

setup_env()

# Load all experiment configurations from local project
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
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict

path = "/Users/sknudstrup/tbp/results/dmc/results/dist_agent_1lm/eval_stats.csv"
df = pd.read_csv(path)
n_episodes = len(df)
vc = dict(df["primary_performance"].value_counts())
print(f"Correct: {100 * vc['correct'] / n_episodes}")
print(f"Correct MLH: {100 * vc['correct_mlh'] / n_episodes}")
print(f"Correct*: {100 * (vc['correct'] + vc['correct_mlh']) / n_episodes}")

path = "/Users/sknudstrup/tbp/results/dmc/results.old/dist_agent_1lm/eval_stats.csv"
df = pd.read_csv(path)
n_episodes = len(df)
vc = dict(df["primary_performance"].value_counts())
print(f"Correct: {100 * vc['correct'] / n_episodes}")
print(f"Correct MLH: {100 * vc['correct_mlh'] / n_episodes}")
print(f"Correct*: {100 * (vc['correct'] + vc['correct_mlh']) / n_episodes}")


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
