from tbp.monty.frameworks.run_env import setup_env

setup_env()

# Load all experiment configurations from local project
from configs import CONFIGS  # noqa: E402
from configs.common import DMC_PRETRAIN_DIR
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.run import main  # noqa: E402

config = CONFIGS["fig3_symmetry_run"]
config["experiment_args"].n_eval_epochs = 1
config["experiment_args"].model_name_or_path = str(
    DMC_PRETRAIN_DIR / "dist_agent_1lm_10distinctobj/pretrained"
)
config["monty_config"].learning_module_configs["learning_module_0"][
    "learning_module_args"
]["use_multithreading"] = False
config.update(
    dict(
        eval_dataloader_class=ED.InformedEnvironmentDataLoader,
        eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
            object_names=["mug"],
            object_init_sampler=PredefinedObjectInitializer(
                rotations=[[196, 326, 225]]
            ),
        ),
    )
)
main(all_configs=CONFIGS, experiments=["fig3_symmetry_run"])
