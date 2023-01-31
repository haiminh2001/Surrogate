import numpy as np
from ...EA import Individual
from ..tasks import create_ZDT

class IND_ZDT(Individual):
    def __init__(self, genes, dim=None, type: int = None):
        super().__init__(genes, dim)
        if genes is None:
            self.genes: np.ndarray = np.random.uniform(0, 1, dim)
            
        self.type = type
    
class ZDT_benchmark:
    def get_tasks(zdt_index = 1):
        
        return create_ZDT(zdt_index), IND_ZDT
    
    
    
