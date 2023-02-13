import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as MAPE

class BaseSingleModel:
    def __init__(self, eval_metric = MAPE):
        self.eval_metric = eval_metric
    
    def fit(self, X, y):
        pass
        
    
    def predict(self, X):
        pass
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        return self.eval_metric(y, y_hat)
        


class BaseSubpopSurrogate:
    def __init__(self, model_cls):
        self.model = model_cls()
        
    def fit(self, X, y):
        raise Exception("Fit function not implemented")
        
    
    def predict(self, X):
        raise Exception("Predict function not implemented")
    
    def evaluate(self, X, y):
        raise Exception("Evaluate function not implemented")

    
class BaseSurrogate:
    def __init__(self, num_sub_pop: int, eval_metric = MAPE, subpop_surroagte_class= BaseSubpopSurrogate, single_modeLclass = BaseSingleModel):
        super().__init__()
        self.models: list = [subpop_surroagte_class() for _ in range(num_sub_pop)]
        self.eval_metric = eval_metric
        self.num_sub_pop = num_sub_pop
    
    def prepare_data(self, skf, genes, costs = None):
        for i in self.num_sub_pop:
            index = skf == i 
            
            if costs == None:
                yield genes[index]
            else:
                yield genes[index], costs[index]
        
    def fit(self, genes, costs, skf):
        
        for i, (X, y) in enumerate(self.prepare_data(skf, genes, costs)):
            if len(y):
                assert len(X) == len(y)
                self.models[i].fit(X, y)
    
    def predict(self, genes, skf):
        for i, X in enumerate(self.prepare_data(skf, genes)):
            if len(X):
                yield self.models[i].predict(X)
            else:
                yield [] 
    
    def evaluate(self, genes, costs, skf):
        for i, (X, y) in enumerate(self.prepare_data(skf, genes, costs)):
            if len(y):
                assert len(X) == len(y)
                self.modesl[i].evaluate(X, y)
            else:
                yield None
    
class MOO_BaseSubpopSurrogate(BaseSubpopSurrogate):
    def __init__(self, num_objs: int, num_dims: list, model_cls = BaseSingleModel, eval_metric = MAPE):
        self.num_objs = num_objs
        self.num_dims = num_dims
        self.models = [model_cls(eval_metric) for _ in range(num_objs)]
        
    def predict(self, X):
        for i, model in enumerate(self.models):
            yield model.predict(X[: self.num_dims[i]])
    
    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.fit(X[: self.num_dims[i]], y[:, i])

    def evaluate(self, X, y):
        for i, model in enumerate(self.models):
            model.evaluate(X[: self.num_dims[i]], y[:, i])

        