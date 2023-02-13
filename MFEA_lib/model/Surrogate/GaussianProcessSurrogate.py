from .BaseSurrogate import BaseSingleModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE


class GaussianProcessSingleModel(BaseSingleModel):
    def __init__(self, eval_metric = MAPE, random_state = 0.42):
        super().__init__(eval_metric= eval_metric)
        self.model = GaussianProcessRegressor(random_state=random_state)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def fit(self, X, y):
        self.model.fit(X, y)



    
