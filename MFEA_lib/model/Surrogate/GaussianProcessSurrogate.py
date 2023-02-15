from .BaseSurrogate import BaseSingleModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import numpy as np

class GaussianProcessSingleModel(BaseSingleModel):
    def __init__(self, eval_metric = MAPE, random_state = 42):
        super().__init__(eval_metric= eval_metric)
        self.model = GaussianProcessRegressor(random_state=random_state)
    
    def predict(self, X:np.ndarray):
        assert type(X) == np.ndarray 
        return self.model.predict(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        assert type(X) == np.ndarray and type(y) == np.ndarray
        self.model.fit(X, y)



    
