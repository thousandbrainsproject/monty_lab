from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from playground.flop_counter import FlopCounter
from frameworks.models.model_base import MontyFlopsBase

# Make the above class inherit from MontyFlopsExperiment
class MontyObjectRecognitionFlopsExperiment(MontyObjectRecognitionExperiment):
    def __init__(self):
        super().__init__()
        self.flop_counter = FlopCounter()

    def __new__(cls, *args, **kwargs):
        # Save the original bases
        original_bases = cls.__bases__

        # Temporarily modify the inheritance to use MontyBaseFlops
        cls.__bases__ = (MontyFlopsBase,)

        # Create instance
        instance = super().__new__(cls)

        # Restore original bases
        cls.__bases__ = original_bases

        return instance

    # def run_episode_steps(self):
    #     # Reset flops before the episode starts
    #     self.flop_counter.flops = 0
    #     with self.flop_counter:
    #         loader_step = super().run_episode_steps()
    #     print(f"FLOPs: {self.flop_counter.flops:,}")
    #     return loader_step
