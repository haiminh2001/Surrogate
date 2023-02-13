from .task import AbstractTask
import numpy as np
import numba as nb
from pymoo.problems import get_problem
import traceback
from termcolor import colored

N_PARETO_POINTS = 100
EPS = 1e-12

ZDT1_PARAMS = {'dim': 30, 'up': 1, 'low': 0, 'optimal': np.zeros(30)
               ,'pareto_front': np.array([np.linspace(0, 1, N_PARETO_POINTS), 1 - np.sqrt(np.linspace(0, 1, N_PARETO_POINTS))]).T
    } 
ZDT2_PARAMS = {'dim': 30, 'up': 1, 'low': 0, 'optimal': np.zeros(30),
               'pareto_front': np.array([np.linspace(0, 1, N_PARETO_POINTS), 1 - np.power(np.linspace(0, 1, N_PARETO_POINTS), 2)]).T} 

regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

pf = []

for r in regions:
    x1 = np.linspace(r[0], r[1], int(N_PARETO_POINTS / len(regions)))
    x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
    pf.append(np.array([x1, x2]).T)
    
ZDT3_PARAMS = {'dim': 30, 'up': 1, 'low': 0, 'optimal': np.zeros(30),
               'pareto_front': np.concatenate(pf)} 

ZDT4_PARAMS = {'dim': 10, 'up': np.concatenate((np.array([1]), 5 * np.ones(9))), 'low':  np.concatenate((np.array([0]), -5 * np.ones(9))),
               'optimal': np.zeros(30),'pareto_front': np.array([np.linspace(0, 1, N_PARETO_POINTS), 1 - np.sqrt(np.linspace(0, 1, N_PARETO_POINTS))]).T
               }

x = 1 + np.linspace(0, 1, N_PARETO_POINTS) * 30
m = 11
ZDT5_PARAMS = {'dim': 80, 'up': 1, 'low':  0, 'optimal': np.zeros(30), 'm': m, 'n': 5, 'normalize': False
               ,'pareto_front': np.column_stack([x, (m-1) / x])}


x = np.linspace(0.2807753191, 1, N_PARETO_POINTS)
ZDT6_PARAMS = {'dim': 10, 'up': 1, 'low':  0, 'optimal': np.zeros(30)
               ,'pareto_front': np.array([x, 1 - np.power(x, 2)]).T}

    
def create_ZDT():
    

    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:], nb.int64), nb.types.UniTuple(nb.float64, 2)(nb.float64[:], nb.int64)])
    def func_zdt1(x, n_var):
        f1 = x[0]
        g = 1 + 9.0 / (n_var - 1) * np.sum(x[1:])
        f2 = g * (1 - np.sqrt((x[0] / g)))
        return f1, f2
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:], nb.int64), nb.types.UniTuple(nb.float64, 2)(nb.float64[:], nb.int64)])
    def func_zdt2(x, n_var):
        
        f1 = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / (n_var - 1)
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        return f1, f2

    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:], nb.int64), nb.types.UniTuple(nb.float64, 2)(nb.float64[:], nb.int64)])
    def func_zdt3(x, n_var):
        
        f1 = x[0]
        c = np.sum(x[1:])
        g = 1.0 + 9.0 * c / (n_var - 1)
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        return f1, f2
    
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:], nb.int64), nb.types.UniTuple(nb.float64, 2)(nb.float64[:], nb.int64)])
    def func_zdt4(x, n_var):
        
        f1 = x[0]

        g = 1.0 + 10 * (n_var - 1)
        
        g = g +  np.sum(x[1:] * x[1:] - 10 * np.cos(4.0 * np.pi * x[1:]))
        h = 1.0 - np.sqrt(f1 / g)
        
        f2 = g * h

        return f1, f2
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:], nb.int64, nb.int64, nb.int64), 
              nb.types.UniTuple(nb.float64, 2)(nb.float64[:], nb.int64, nb.int64, nb.int64)])
    def func_zdt5(x, n_var, m, n):
        
        _x = np.empty((m - 1, n), dtype = np.float64)
        for i in nb.prange(m - 1):
            _x[i] = x [30 + i * n: 30 + (i + 1) * n]
        
        # u = np.concatenate((np.array([np.sum(x[:30])]),np.sum(_x, axis = 1)))
        u = np.sum(_x, axis = 1)
        v = (2 + u) * (u < n) + 1 * (u == n)
        g = np.sum(v)

        f1 = 1 + np.sum(x[:30])
        f2 = g * (1 / f1)
        return f1, f2
    
    @nb.njit([nb.types.UniTuple(nb.float64, 2)(nb.int64[:], nb.int64), 
              nb.types.UniTuple(nb.float64, 2)(nb.float64[:], nb.int64)])
    def func_zdt6(x, n_var):
        
        f1 = 1 - np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6)
        g = 1 + 9.0 * np.power(np.sum(x[1:]) / (n_var - 1.0), 0.25)
        f2 = g * (1 - np.power(f1 / g, 2))
        
        return f1, f2
    
    tasks = []
    
    num_tasks = 0
    for k, v in locals().items():
        if 'func_zdt' in k:
            tasks.append(ZDT_Task(**globals()[f'ZDT{num_tasks + 1}_PARAMS'], func = v, test_id = num_tasks + 1))
            num_tasks += 1
            
    assert num_tasks == 6, num_tasks
    return tasks
    

class ZDT_Task(AbstractTask):
    def __init__(self, dim:int, up, low, optimal, pareto_front, func, test_id, m = None, n = None, normalize = None):
        self.dim = dim 
        self.up = up if type(up) == np.ndarray else np.full(dim, up)
        self.low = low if type(low) == np.ndarray else np.full(dim, low)
        self.func = func
        self.optimal = optimal 
        self.pareto_front = pareto_front
        
        if normalize is not None:
            self.pymoo_func = get_problem(f'zdt{test_id}', normalize = normalize)    
        else:
            self.pymoo_func = get_problem(f'zdt{test_id}')
        self.test_id = f'problem {test_id}'
        self.args = [arg for arg in [dim, m, n] if arg]
        self.test_value()
    
    @property
    def dims(self):
        return np.array([self.dim] * self.num_objs)
    
    @property
    def num_objs(self):
        return 2
         

        
    
    def test_value(self):
        try:
            #test evaluate
            x = np.stack([np.random.uniform(low=self.low, high=self.up) for _ in range(20)])
            
            pymoo_val = self.pymoo_func.evaluate(x)
            our_val = np.stack([self(x_i) for x_i in x])
            
            assert our_val.shape[0] == pymoo_val.shape[0] and pymoo_val.shape[1] == our_val.shape[1], (our_val.shape, pymoo_val.shape)
            
            diff = np.abs((our_val - pymoo_val) / (pymoo_val + 1))
            assert np.all(diff < EPS), (our_val[:10], pymoo_val[:10])
            
            #test paretofront
            pymoo_pareto_front = self.pymoo_func.pareto_front()
            assert self.pareto_front.shape[0] == pymoo_pareto_front.shape[0]
            
            
            diff = np.abs((self.pareto_front - pymoo_pareto_front) / (pymoo_pareto_front + 1))
            assert np.all(diff < EPS), (self.pareto_front[:10], pymoo_pareto_front[:10])
        except:
            print(colored(f'========{self.test_id}===========', 'red'))
            print(traceback.format_exc())
            print(colored(f'============================', 'red'))
            
    
    def __call__(self, X: np.ndarray):

        return self.func(np.clip(X[:self.dim], self.low, self.up), *self.args)

