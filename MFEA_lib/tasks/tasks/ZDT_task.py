from .task import AbstractTask
from pymoo.problems import get_problem


import numpy as np
class ZDT_Task(AbstractTask):
    def __init__(self, id:int ):
        
        self.problem = globals()[f'ZDT{id}']
        self.dim: int = self.problem.n_var
        self.up: int = self.problem.xu
        self.low: int = self.problem.xl
        self.num_object: int = self.problem.n_obj
        
        
    
    def __call__(self, X: np.ndarray):
        return self.problem.evaluate(X[:self.dim])
    
class ZDT:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v) 
            
        #create tasks list
        for k, v in self.__dict__:
            if 'evaluate_func' in k:
                pass
        # n_var=n_var, n_obj=2, xl=0, xu=1, vtype=float
    
class ZDT1(ZDT):
    def __init__(self):
        super().__init__(n_var = 30, n_obj = 2, xl = 0, xu = 1)
        

    def _evaluate_func_1(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        


class ZDT2(ZDT):



    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        out["F"] = np.column_stack([f1, f2])


class ZDT3(ZDT):


    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        out["F"] = np.column_stack([f1, f2])


class ZDT4(ZDT):
    def __init__(self, n_var=10):
        super().__init__(n_var)
        self.xl = -5 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 5 * np.ones(self.n_var)
        self.xu[0] = 1.0
        self.func = self._evaluate


    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * np.cos(4.0 * np.pi * x[:, i])
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h

        out["F"] = np.column_stack([f1, f2])


class ZDT5(ZDT):

    def __init__(self, m=11, n=5, **kwargs):
        self.m = m
        self.n = n
        super().__init__(n_var=(30 + n * (m - 1)), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(float)

        _x = [x[:, :30]]
        for i in range(self.m - 1):
            _x.append(x[:, 30 + i * self.n: 30 + (i + 1) * self.n])

        u = np.column_stack([x_i.sum(axis=1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[:, 1:].sum(axis=1)

        f1 = 1 + u[:, 0]
        f2 = g * (1 / f1)


        out["F"] = np.column_stack([f1, f2])


class ZDT6(ZDT):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)


    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1 - np.exp(-4 * x[:, 0]) * np.power(np.sin(6 * np.pi * x[:, 0]), 6)
        g = 1 + 9.0 * np.power(np.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
        f2 = g * (1 - np.power(f1 / g, 2))

        out["F"] = np.column_stack([f1, f2])