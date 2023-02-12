
class BaseSurrogate:
    def __init__(self):
        pass
    
    def prepare_data(self, genes, costs, skf):
        raise Exception("Prepare data function not implemented")
        
    def fit(self, X, y):
        raise Exception("Fit function not implemented")
        
    
    def predict(self, X):
        raise Exception("Predict function not implemented")
    
    def evaluate(self, X, y):
        raise Exception("Evaluate function not implemented")