import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as MAPE

class BaseSingleModel:
    def __init__(self, eval_metric = MAPE, init_before_fit = False):
        self.eval_func = eval_metric
        self.init_before_fit = init_before_fit
    
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
    def __init__(self, single_model_class):
        self.model = single_model_class()
        
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
                 single_model_class = BaseSingleModel.__class__):
        super().__init__()
        self.models: list = [subpop_surroagte_class(eval_metric = eval_metric, single_model_class= single_model_class) for _ in range(num_sub_pop)]
        self.eval_metric = eval_metric
        self.num_sub_pop = num_sub_pop
        self.is_init = False
    
    def init_subpop_models(self, num_objs:list, dims: np.ndarray):
        for i, model in enumerate(self.models):
            model.init_single_model(num_objs= num_objs[i], dims= dims[i])
        self.is_init = True
        
    
    def prepare_data(self, skf, genes, costs):
        assert self.is_init, 'Surrogate model not initialized!!'
        for i in range(self.num_sub_pop):
            index = skf == i 
            
            if type(costs) != np.ndarray:
                yield genes[index]
            else:
                yield genes[index], costs[index]
        
    def fit(self, genes, costs, skf):
        assert self.is_init, 'Surrogate model not initialized!!'
        assert len(costs.shape) >= 2, costs.shape
        for i, (X, y) in enumerate(self.prepare_data(skf, genes, costs)):
            if len(y):
                assert len(X) == len(y)
                self.models[i].fit(X, y)
    
    def predict(self, genes, skf):
        assert self.is_init, 'Surrogate model not initialized!!'
        for i, X in enumerate(self.prepare_data(skf, genes)):
            if len(X):
                yield self.models[i].predict(X)
            else:
                yield [] 
    
    def evaluate(self, genes, costs, skf):
        assert self.is_init, 'Surrogate model not initialized!!'
        for i, (X, y) in enumerate(self.prepare_data(skf, genes, costs)):
            if len(y):
                assert len(X) == len(y)
                yield self.models[i].evaluate(X, y)
            else:
                yield None
    
class MOO_BaseSubpopSurrogate(BaseSubpopSurrogate):
    def __init__(self, single_model_class = BaseSingleModel.__class__, eval_metric = MAPE):
        self.single_model_class = single_model_class
        self.eval_metric = eval_metric
        
    
    def init_single_model(self, num_objs: int, dims: np.ndarray):
        assert len(dims.shape) == 1, 'dims must be an array'
        self.num_objs = num_objs
        self.dims = dims
        self.models = [self.single_model_class(self.eval_metric) for _ in range(num_objs)]
        
    def predict(self, X):
        for i, model in enumerate(self.models):
            yield model.predict(X[:, : self.dims[i]])
    
    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X[:, : self.dims[i]], y[:, i])

    def evaluate(self, X, y):
        result = []
        for i, model in enumerate(self.models):
            result.append(model.evaluate(X[:, : self.dims[i]], y[:, i]))
        return result
        