from .fig3_robust_sensorimotor_inference import CONFIGS as FIG_3_CONFIGS
from .fig4_structured_object_representations import CONFIGS as FIG_4_CONFIGS
from .fig5_rapid_inference_with_voting import CONFIGS as FIG_5_CONFIGS
from .fig6_rapid_inference_with_model_based_policies import CONFIGS as FIG_6_CONFIGS
from .fig7_rapid_learning import CONFIGS as FIG_7_CONFIGS
from .fig8_flops import CONFIGS as FIG_8_CONFIGS
from .pretraining_experiments import CONFIGS as PRETRAINING_CONFIGS
from .view_finder_images import CONFIGS as VIEW_FINDER_CONFIGS
from .visualizations import CONFIGS as VISUALIZATION_CONFIGS

CONFIGS = dict()
CONFIGS.update(PRETRAINING_CONFIGS)
CONFIGS.update(FIG_3_CONFIGS)
CONFIGS.update(FIG_4_CONFIGS)
CONFIGS.update(FIG_5_CONFIGS)
CONFIGS.update(FIG_6_CONFIGS)
CONFIGS.update(FIG_7_CONFIGS)
CONFIGS.update(FIG_8_CONFIGS)
CONFIGS.update(VIEW_FINDER_CONFIGS)
CONFIGS.update(VISUALIZATION_CONFIGS)