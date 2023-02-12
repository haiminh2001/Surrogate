from .BaseSurrogate import BaseSurrogate
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as MAPE
class GaussianProcessSurrogate(BaseSurrogate):
    def __init__(self, eval_metric = MAPE):
        super().__init__()
        self.model: GaussianProcessRegressor = GaussianProcessRegressor()
        self.eval_metric = eval_metric
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_hat = self.model.predict
        return self.eval_metric(y, y_hat)
    
    def prepare_data(self, genes, costs, skf):
        return super().prepare_data(genes, costs, skf)