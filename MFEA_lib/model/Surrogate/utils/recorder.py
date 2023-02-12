import numpy as np

class BaseRecorder:
    def __init__(self):
        self.genes = []
        self.costs = []
        self.skf = []
        self.last_genes = None
        self.last_costs = None
        self.last_skf = None
    
    def __open__(self):
        pass
    
    def __exit__(self):
        pass
    
    def record(self, genes, costs, skf):
        pass
    
    @property
    def getall(self):
        return self.genes, self.costs, self.skf 
    
    @property
    def getlast(self):
        return self.last_genes, self.last_costs, self.last_skf
    

class InMemRecorder(BaseRecorder):
    def __init__(self):
        super().__init__()
        
    def record(self, genes, costs, skf):
        self.genes.extend(genes)
        self.costs.extend(costs)
        self.skf.extend(skf)
        self.last_genes = genes
        self.last_costs = costs
        self.last_skf = skf
    
class InMemRecorderNumpy(BaseRecorder):
    def __init__(self):
        super().__init__()
        
    def record(self, genes, costs, skf):
        self.genes.extend(genes)
        self.costs.extend(costs)
        self.skf.extend(skf)
        self.last_genes = genes
        self.last_costs = costs
        self.last_skf = skf
    
    @property
    def getall(self):
        return np.array(self.genes), np.array(self.costs), np.array(self.skf)
    
    @property
    def getlast(self):
        return np.array(self.last_genes), np.array(self.last_costs), np.array(self.last_skf)