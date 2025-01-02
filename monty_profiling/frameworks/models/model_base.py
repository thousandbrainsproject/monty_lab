from tbp.monty.frameworks.models.monty_base import MontyBase


class MontyFlopsBase(MontyBase):
    def __init__(self):
        super().__init__()
        self.flop_counter = FlopCounter()

    def step(self, observation):
        with self.flop_counter:
            super().step(observation)
        print(f"FLOPs: {self.flop_counter.flops:,}")
