from MFEA_lib.tasks.Benchmark.zdt import ZDT_benchmark
from MFEA_lib.model import MFEA_base
from MFEA_lib.model.utils import *
from MFEA_lib.operators.Crossover import SBX_Crossover
from MFEA_lib.operators.Mutation import *
from MFEA_lib.operators.Selection import *
from datetime import datetime
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--use_surrogate', action = 'store_true', help= 'use surrogate or not')
    parser.add_argument('--device', type = str, default= 'cuda', help= 'device')
    parser.add_argument('--record', action = 'store_true', help= 'record gene and its fitness')
    parser.add_argument('--task', type = int, default= 1, help= 'choose task')
    parser.add_argument('--num_loops', type = int, default= 1, help= 'choose task')
    parser.add_argument('--train_period', type = int, default= 5, help= 'train period')
    parser.add_argument('--start_eval', type = int, default= 5, help= 'start eval')
    parser.add_argument('--save_path', type = str, default= "")
    parser.add_argument('--merge', action = 'store_true', help= 'merge data of current task')
    return parser

def main():
    
    args = get_parser().parse_args()
    kwargs = {}
    tasks, IndClass = ZDT_benchmark.get_tasks()
    
    # t = tasks[0]
    # a = np.array([1] * 30).astype(np.int64)
    # print(type(t))
    # print(t(a))
        
    for loop in range(args.num_loops):
      baseModel = MFEA_base.betterModel()
      baseModel.compile(
          IndClass= IndClass,
          tasks= tasks,        
          crossover= SBX_Crossover(),
          mutation= NoMutation(),
          selection= ElitismSelection(),
          record = args.record,
          merge = args.merge,
          **kwargs
      )
      solve = baseModel.fit(
          nb_generations = 100, rmp = 0.5, nb_inds_each_task= 100, 
          bound_pop= [0, 1], evaluate_initial_skillFactor= False, is_moo = True,
          train_period = args.train_period, start_eval = args.start_eval,
      )

if __name__ == '__main__':
    main()
