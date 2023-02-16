import numpy as np
from typing import Type

class BaseRecorder:
    def __init__(self, subset_selection:Type = None, test_amount = 0.1):
        self.genes = []
        self.costs = []
        self.skf = []
        self.last_genes = None
        self.last_costs = None
        self.last_skf = None
        if subset_selection:
            self.subset_selection = subset_selection()
        self.test_amount = test_amount
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.save_data()
    
    def save_data(self):
        pass
    
    def record(self, genes, costs, skf):
        pass
    
    @property
    def all(self):
        return self.genes, self.costs, self.skf 
    
    @property
    def last(self):
        return self.last_genes, self.last_costs, self.last_skf
    
    @property
    def last_train_test_split(self):
        self.subset_selection.set_subset(self.last, 1 - self.test_amount)
        return self.subset_selection.train_inds, self.subset_selection.test_inds
    

class InMemRecorder(BaseRecorder):
    def __init__(self, subset_selection:Type = None, test_amount = 0.1):
        super().__init__(subset_selection, test_amount)
        
    def record(self, genes, costs, skf, offspring):
        self.genes.extend(genes)
        self.costs.extend(costs)
        self.skf.extend(skf)
        self.last_genes = genes
        self.last_costs = costs
        self.last_skf = skf
        self.last_offspring = offspring
    
class InMemRecorderNumpy(InMemRecorder):
    def __init__(self, subset_selection:Type = None, test_amount = 0.1):
        super().__init__(subset_selection, test_amount)
            
    @property
    def all(self):
        return np.array(self.genes), np.array(self.costs), np.array(self.skf)
    
    @property
    def last(self):
        return np.array(self.last_genes), np.array(self.last_costs), np.array(self.last_skf)
    
    @property
    def last_train_test_split(self):
        self.subset_selection.set_subset(self.last_offspring, 1 - self.test_amount)
        return [np.array(x) for x in self.subset_selection.train_inds], [np.array(x) for x in  self.subset_selection.test_inds]