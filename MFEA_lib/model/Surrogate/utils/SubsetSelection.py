from MFEA_lib.EA import Population
import numpy as np
class BaseSubsetSelection:
    def __init__(self):
        self.subset_train: Population = None
        self.subset_test: Population = None
    
    def set_subset(self, population: Population, train_amount = 0.5):
        assert train_amount >= 0 and train_amount <=1
        self.subset_train = Population(
            population.IndClass,
            nb_inds_tasks = [0] * population.nb_tasks, 
            dim = population.dim_uss,
            list_tasks= population.ls_tasks,
            is_moo = population.is_moo
        )
        
        self.subset_test = Population(
            population.IndClass,
            nb_inds_tasks = [0] * population.nb_tasks, 
            dim = population.dim_uss,
            list_tasks= population.ls_tasks,
            is_moo = population.is_moo
        )
        
        
        
        for i, subpop in population:
            #shuffle 
            perm = np.random.permutation(len(subpop))
            train_index = perm[:int(len(subpop) * train_amount)]
            test_index = perm[int(len(subpop) * train_amount):]
            
            self.subset_train[i] = self.subset_train[i] + subpop[train_index]
            self.subset_test[i] = self.subset_test[i] + subpop[test_index]
    
    @property
    def train_inds(self) -> list:
        return self.subset_train.get_all_inds()
        
    @property
    def test_inds(self) -> list:
        return self.subset_test.get_all_inds()