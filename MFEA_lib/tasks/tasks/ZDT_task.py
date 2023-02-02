from .task import AbstractTask
import numpy as np
import numba as nb
from pymoo.problems import get_problem

ZDT1_PARAMS = {'dim': 30, 'up': 1, 'low': 0} 
ZDT2_PARAMS = {'dim': 30, 'up': 1, 'low': 0} 
ZDT3_PARAMS = {'dim': 30, 'up': 1, 'low': 0} 
ZDT4_PARAMS = {'dim': 10, 'up': np.concatenate((np.array([1]), 5 * np.ones(9))), 'low':  np.concatenate((np.array([0]), -5 * np.ones(9)))}
ZDT5_PARAMS = {'dim': 80, 'up': 1, 'low':  0}
ZDT6_PARAMS = {'dim': 10, 'up': 1, 'low':  0}

    
def create_ZDT():
    

    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:]), nb.types.UniTuple(nb.float64, 2)(nb.float64[:])])
    def func_zdt1(x):
        f1 = x[0]
        g = 1 + 9.0 / 29 * np.sum(x[1:])
        f2 = g * (1 - np.sqrt((x[0] / g)))
        return f1, f2
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:]), nb.types.UniTuple(nb.float64, 2)(nb.float64[:])])
    def func_zdt2(x):
        
        f1 = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / 29
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        return f1, f2

    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:]), nb.types.UniTuple(nb.float64, 2)(nb.float64[:])])
    def func_zdt2(x):
        
        f1 = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / 29
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        return f1, f2
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:]), nb.types.UniTuple(nb.float64, 2)(nb.float64[:])])
    def func_zdt3(x):
        
        f1 = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / 29
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        return f1, f2
    
    #NOTE: implement zdt 5, 6
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:]), nb.types.UniTuple(nb.float64, 2)(nb.float64[:])])
    def func_zdt4(x):
        
        f1 = x[0]
    
        
        g = 91 +  np.sum(x[1:] * x[1:] - 10 * np.cos(4.0 * np.pi * x[1:]))
        h = 1.0 - np.sqrt(f1 / g)
        
        f2 = g * h

        return f1, f2
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:]), nb.types.UniTuple(nb.float64, 2)(nb.float64[:])])
    def func_zdt5(x):
        
        f1 = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / 29
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        return f1, f2
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:]), nb.types.UniTuple(nb.float64, 2)(nb.float64[:])])
    def func_zdt6(x):
        
        f1 = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / 29
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        return f1, f2
    
    tasks = []
    
    num_tasks = 0
    for k, v in locals().items():
        if 'func_zdt' in k:
            tasks.append(ZDT_Task(**globals()[f'ZDT{num_tasks + 1}_PARAMS'], func = v))
            num_tasks += 1
            
    assert num_tasks == 6, num_tasks
    return tasks
    

class ZDT_Task(AbstractTask):
    def __init__(self, dim:int, up:int, low:int, func):
        self.dim = dim 
        self.up = up 
        self.low = low
        self.func = func
        
    
    def __call__(self, X: np.ndarray):
        return self.func(np.clip(X[:self.dim], self.low, self.up))

