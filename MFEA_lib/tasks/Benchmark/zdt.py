import numpy as np
from ...EA import Individual
from ..tasks import create_ZDT

class IND_ZDT(Individual):
    def __init__(self, genes, dim=None, type: int = None):
        super().__init__(genes, dim)
        if genes is None:
            self.genes: np.ndarray = np.random.uniform(-5, 5, dim)
            
        self.type = type
    
class ZDT_benchmark:
    def get_tasks():        
        return create_ZDT(), IND_ZDT
    
    
    
