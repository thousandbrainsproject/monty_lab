from tbp.monty.frameworks.run_env import setup_env

setup_env()

# Load all experiment configurations from local project
from configs import CONFIGS  # noqa: E402
from tbp.monty.frameworks.run import main  # noqa: E402

experiment_name = "fig6_surf_mismatch"
config = CONFIGS[experiment_name]

main(all_configs=CONFIGS, experiments=[experiment_name])
