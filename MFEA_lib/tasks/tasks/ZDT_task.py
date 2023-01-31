from .task import AbstractTask
import numpy as np
import numba as nb
def create_ZDT(id):
    return globals()[f'create_ZDT{id}']()
    
    
def create_ZDT1():
    params = {'dim': 30, 'up': 1, 'low': 0}
    
    @nb.njit([nb.float64(nb.int64[:]), nb.float64(nb.float64[:])])
    def func1(x):
        return np.float64(x[0])
    
    @nb.njit([nb.float64(nb.int64[:]), nb.float64(nb.float64[:])])
    def func2(x):
        g = 1 + 9.0 / 29 * np.sum(x)
        return g * (1 - np.sqrt((x[0] / g)))
    
    return ZDT_Task(**params, func = func1), ZDT_Task(**params, func = func2)

def create_ZDT2():
    params = {'dim': 30, 'up': 1, 'low': 0}
    
    @nb.njit([nb.float64(nb.int64[:]), nb.float64(nb.float64[:])])
    def func1(x):
        return np.float64(x[0])
    
    @nb.njit([nb.float64(nb.int64[:]), nb.float64(nb.float64[:])])
    def func2(x):
        g = 1 + 9.0 / 29 * np.sum(x[:, 1:], axis=1)
        return g * (1 - np.sqrt((x[0] / g)))
    
    return ZDT_Task(**params, func = func1), ZDT_Task(**params, func = func2)
    

class ZDT_Task(AbstractTask):
    def __init__(self, dim:int, up:int, low:int, func):
        self.dim = dim 
        self.up = up 
        self.low = low
        self.func = func
        
    
    def __call__(self, X: np.ndarray):
        return self.func(X[:self.dim])

