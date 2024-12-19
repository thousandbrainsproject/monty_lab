from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from playground.flop_counter import FlopCounter


# Make the above class inherit from MontyFlopsExperiment
class MontyObjectRecognitionFlopsExperiment(MontyObjectRecognitionExperiment):
    def __init__(self):
        super().__init__()
        self.flop_counter = FlopCounter()

    def run_episode(self):
        # Reset flops before the episode starts
        self.flop_counter.flops = 0
        with self.flop_counter:
            super().run_episode()
        print(f"FLOPs: {self.flop_counter.flops}")
