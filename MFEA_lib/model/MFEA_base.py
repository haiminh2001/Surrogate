import numpy as np
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..EA import *
from .Surrogate.utils import BaseRecorder, BaseSubsetSelection
from termcolor import colored
from .Surrogate import BaseSurrogate
from typing import Optional, Type


class betterModel(AbstractModel.model):
    def compile(self, 
        IndClass: Type,
        tasks: list, 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.Polynomial_Mutation, selection: Selection.ElitismSelection, 
        record: bool = False, 
        merge: bool = False, 
        save_path:str = None,
        use_surrogate: bool = False,
        surrogate_model: BaseSurrogate = Optional,
        recorder_class: Type(BaseRecorder) = Optional,
        subset_selection: BaseSubsetSelection = Optional,
        surrogate_params: dict = {},
        *args, **kwargs):
        
        if use_surrogate:
            self.surrogate_model = surrogate_model
            self.surrogate_params = surrogate_params
            self.recorder_class = recorder_class
            self.use_surrogate = use_surrogate
            self.subset_selection = subset_selection
            dims = np.array([task.dims for task in tasks])
            num_objs = [task.num_objs for task in tasks]
            self.surrogate_model.init_subpop_models(num_objs=num_objs, dims= dims)            
            
        self.record = record
        self.merge = merge
        self.save_path  = save_path
        self.update_history = None
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
    
    def fit(self, nb_generations, rmp = 0.3, nb_inds_each_task = 100,
            evaluate_initial_skillFactor = True, init_surrogate_gens = 5
            , start_eval = 6, is_moo = False, *args, **kwargs) -> list:
        super().fit(*args, **kwargs)
        
        # initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor,
            is_moo = is_moo
        )
      
        # save history
        def update_non_moo():
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
        
        def update_moo():
            self.history_cost.append([subpop.igd for subpop in population])
        
        self.update_history = update_moo if is_moo else update_non_moo
        self.update_history()

        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys= True)
                    
        with self.recorder_class() as recorder:
            for epoch in range(nb_generations):
                #evo step
                genes, costs, skf, population= self.epoch_step(rmp, epoch, nb_inds_each_task, nb_generations, population)
                if self.use_surrogate:
                    recorder.record(genes, costs, skf)
                
                if epoch == init_surrogate_gens - 1:
                    genes, costs, skf = recorder.getall
                    self.surrogate_model.fit(genes, costs, skf)
                

        self.write_data()
        
        print(colored('\nEND!', 'red'))

        #solve
        self.last_pop = population
        return self.last_pop.get_solves() 
    
    
    def epoch_step(self, rmp, cur_epoch, nb_inds_each_task, nb_generations, population):
        # initial offspring_population of generation
        offsprings = Population(
            self.IndClass,
            nb_inds_tasks = [0] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            is_moo = population.is_moo
        )
        

        # create offspring pop
        while len(offsprings) < len(population):
            # choose parent 
            pa, pb = population.__getRandomInds__(2)

            if pa.skill_factor == pb.skill_factor or np.random.rand() < rmp:
                # intra / inter crossover
                skf_oa, skf_ob = np.random.choice([pa.skill_factor, pb.skill_factor], size= 2, replace= True)
                oa, ob = self.crossover(pa, pb, skf_oa, skf_ob)
            else:
                # mutate
                oa = self.mutation(pa, return_newInd= True)
                oa.skill_factor = pa.skill_factor

                ob = self.mutation(pb, return_newInd= True)    
                ob.skill_factor = pb.skill_factor
            
            offsprings.__addIndividual__(oa)
            offsprings.__addIndividual__(ob)

        #select subset for surrogate
        if self.use_surrogate:
            self.subset_selection.set_subset(offsprings, train_amount= 0.1)
            subset_inds = self.subset_selection.inds
        
        # merge and update rank
        population = population + offsprings
        population.update_rank()
        
        # selection
        self.selection(population, [nb_inds_each_task] * len(self.tasks))
        
        # save history
        self.update_history()

        #print
        self.render_process((cur_epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        return np.stack([ind.genes for ind in subset_inds]), np.stack([ind.fcost for ind in subset_inds]), \
                np.stack([ind.skill_factor for ind in subset_inds]), population
                


    