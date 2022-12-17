from torch_geometric.data import Data, InMemoryDataset
import torch
import numpy as np
from tqdm import tqdm
class stacking_data(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        return None


class GraphDataset(InMemoryDataset):
    def __init__(self, tasks, root = './data', transform= None, pre_transform= None, pre_filter= None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = []
        self.tasks= tasks
        self.latest_data = []
       
    @staticmethod 
    def create_new_graph_instance(edge_index, edge_attribute, num_nodes: int, source:int, target:int, 
                                  genes: np.ndarray, y: int, thresh_hold: int = None,
                                  skill_factor: int = None, max_num_domains: int = None,
                                  max_num_nodes:int = None):
        idx = np.sort(np.argsort(genes[0])[:num_nodes])
        onehot_prior = np.zeros((max_num_nodes, max_num_nodes))
        onehot_k = np.zeros((max_num_nodes, max_num_domains))
        for j, i in enumerate(idx): 
          onehot_prior[j, genes[0, i] ] = 1
          onehot_k[j, genes[1, i]] = 1
        
        x = np.zeros((max_num_nodes, 2))
        x[source][0] = 1
        x[target][1] = 1
        x = np.hstack((x, onehot_prior, onehot_k))

        x = torch.tensor(x, dtype= torch.float)
        return dict(x= x, edge_index= edge_index, edge_attr= edge_attribute, y= torch.Tensor([y]),
                    thresh_hold = torch.Tensor([1 if y < thresh_hold else 0]), skill_factor = torch.Tensor([skill_factor]))
    
    def set_max_min(self, x_max, x_min, edge_max, edge_min):
      self.x_max, self.x_min, self.edge_max, self.edge_min = x_max, x_min, edge_max, edge_min

    def append(self, genes, costs, skfs, thresh_hold, max_num_domains, max_num_nodes):
        self.latest_data = []
        for i in tqdm(range(genes.shape[0])):
            self.latest_data.append(__class__.create_new_graph_instance(
                edge_index = self.tasks[skfs[i]].edge_index,
                edge_attribute= self.tasks[skfs[i]].edge_attribute,
                num_nodes= self.tasks[skfs[i]].num_nodes, 
                source= self.tasks[skfs[i]].source,
                target= self.tasks[skfs[i]].target,
                genes = genes[i], 
                y = costs[i],
                thresh_hold= thresh_hold[i],
                skill_factor = skfs[i],
                max_num_domains = max_num_domains,
                max_num_nodes = max_num_nodes,
            ))
        

        self.data.extend(self.latest_data)
        all_data = torch.cat([self.data[i]['x'] for i in range(len(self))])
        self.x_min = all_data.min(dim = 0)[0]
        self.x_max = all_data.max(dim = 0)[0]

        del all_data 
        all_data = torch.cat([self.data[i]['edge_attr'] for i in range(len(self))])
        self.edge_min = all_data.min(dim = 0)[0]
        self.edge_max = all_data.max(dim = 0)[0]

    def len(self):
        return len(self.data)
    
    def normalize(self, data):
      norm_data = dict(**data)
      norm_data['x'] = (norm_data['x'] - self.x_min) / (self.x_max - self.x_min + 1e-5)
      norm_data['edge_attr'] = (norm_data['edge_attr'] - self.edge_min) / (self.edge_max - self.edge_min + 1e-5)
      return norm_data

    def get(self, idx):
        return Data(**self.data[idx])
        
    def _download(self):
        pass
    
    def _process(self):
        pass
    
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__)