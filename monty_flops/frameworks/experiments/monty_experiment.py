from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from playground.flop_counter import FlopCounter
import logging


class MontyFlopsExperiment(MontyExperiment):
    def __init__(self):
        super().__init__()
        self.flop_counter = FlopCounter()

    def run_episode(self):
        """Run one episode until model.is_done."""
        self.pre_episode()
        for step, observation in enumerate(self.dataloader):
            self.pre_step(step, observation)
            with self.flop_counter:
                self.model.step(observation)
                logging.info(f"FLOPs: {self.flop_counter.flops}")
                print(f"FLOPs: {self.flop_counter.flops}")
            self.post_step(step, observation)
            if self.model.is_done or step >= self.max_steps:
                break
        self.post_episode(step)
