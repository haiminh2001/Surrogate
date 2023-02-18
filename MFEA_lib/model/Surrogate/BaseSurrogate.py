import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from typing import Type
class BaseSingleModel:
    def __init__(self, eval_metric = MAPE, retrain_all_data = False):
        self.eval_func = eval_metric
        self.retrain_all_data = retrain_all_data
    
    def fit(self, X, y):
        pass
        
    def init_model(self):
        pass
    
    def predict(self, X):
        pass
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        return self.eval_func(y, y_hat)
        


class BaseSubpopSurrogate:
    def __init__(self, single_model_class, retrain_all_data = False):
        self.model = single_model_class(retrain_all_data)
        
    def init_single_model(self, num_objs: int, dims: np.ndarray):
        pass
        
    def fit(self, X, y):
        raise Exception("Fit function not implemented")
        
    
    def predict(self, X):
        raise Exception("Predict function not implemented")
    
    def evaluate(self, X, y):
        raise Exception("Evaluate function not implemented")

    
class BaseSurrogate:
    def __init__(self, num_sub_pop: int, eval_metric = MAPE, subpop_surroagte_class= BaseSubpopSurrogate.__class__,
                 single_model_class = Type, retrain_all_data:bool = False):
        super().__init__()
        self.models: list = [subpop_surroagte_class(eval_metric = eval_metric, single_model_class= single_model_class, retrain_all_data = retrain_all_data) for _ in range(num_sub_pop)]
        self.eval_metric = eval_metric
        self.num_sub_pop = num_sub_pop
        self.is_init = False
        self.retrain_all_data = retrain_all_data
    
    def init_subpop_models(self, num_objs:list, dims: np.ndarray):
        for i, model in enumerate(self.models):
            model.init_single_model(num_objs= num_objs[i], dims= dims[i])
        self.is_init = True
        
    
    def prepare_data(self, population, get_cost, get_pseudo = False):
        assert self.is_init, 'Surrogate model not initialized!!'
        inds = population.get_all_inds()
        genes = np.stack([ind.genes for ind in inds])
        skf = np.stack([ind.skill_factor for ind in inds])
        
        if get_pseudo:
            costs = np.stack([ind.fcost_for_eval for ind in inds]) 
          
        if get_cost:
            costs = np.stack([ind.fcost for ind in inds]) 
          
            
        for i in range(self.num_sub_pop):
            index = skf == i 
            
            if get_cost or get_pseudo:
                yield genes[index], costs[index], index
            else:
                yield genes[index], index
        
    def fit(self, population):
        assert self.is_init, 'Surrogate model not initialized!!'
        
        for i, (X, y, index) in enumerate(self.prepare_data(population, get_cost= True)):
            if len(y):
                assert len(X) == len(y)
                self.models[i].fit(X, y)
    
    def predict(self, population):
        assert self.is_init, 'Surrogate model not initialized!!'
        #NOTE: hard code dim 1 = 2
        rs = np.empty((len(population), 2), dtype = np.float64)
        for i, (X, index) in enumerate(self.prepare_data(population, get_cost= False)):
            assert len(X)
            rs[index] = np.array(self.models[i].predict(X)).T
        
        return rs
    
    def evaluate(self, population):
        assert self.is_init, 'Surrogate model not initialized!!'
        population.pseudo_evaluate()
        rs = np.empty((len(population), 2), dtype = np.float64)
        for i, (X, y, index) in enumerate(self.prepare_data(population, get_cost = False, get_pseudo = True)):
            assert len(X) == len(y)
            rs[index] =  np.array(self.models[i].evaluate(X, y)).T
        return rs
    
class MOO_BaseSubpopSurrogate(BaseSubpopSurrogate):
    def __init__(self, single_model_class: Type, eval_metric = MAPE, retrain_all_data = False):
        self.single_model_class = single_model_class
        self.eval_metric = eval_metric
        self.retrain_all_data = retrain_all_data
        
    
    def init_single_model(self, num_objs: int, dims: np.ndarray):
        assert len(dims.shape) == 1, 'dims must be an array'
        self.num_objs = num_objs
        self.dims = dims
        self.models = [self.single_model_class(self.eval_metric, retrain_all_data = self.retrain_all_data) for _ in range(num_objs)]
        
    def predict(self, X):
        rs = []
        for i, model in enumerate(self.models):
            rs.append( model.predict(X[:, : self.dims[i]]))
        return rs
    
    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X[:, : self.dims[i]], y[:, i])

    def evaluate(self, X, y):
        result = []
        for i, model in enumerate(self.models):
            result.append(model.evaluate(X[:, : self.dims[i]], y[:, i]))
        return result
        