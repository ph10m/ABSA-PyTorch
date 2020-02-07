import random

from numpy import random as np_random
from torch import cuda, manual_seed
from torch.backends import cudnn


class Seeder:
    def __init__(self, seed=None):
        if seed is None:
            raise 'No seed provided'
        self.seed = seed

    def activate(self):
        random.seed(self.seed)
        np_random.seed(self.seed)
        manual_seed(self.seed)
        cuda.manual_seed(self.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
