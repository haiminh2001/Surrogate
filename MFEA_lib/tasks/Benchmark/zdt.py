import numpy as np
from ...EA import Individual


class IND_ZDT(Individual):
    def __init__(self, genes, dim=None, type: int = None):
        super().__init__(genes, dim)
        if genes is None:
            self.genes: np.ndarray = np.append([np.random.permutation(dim)], [np.random.randint(0, dim, dim)], axis= 0)
            
        self.type = type
    
class ZDT_benchmark:
    def get_task():
        pass
