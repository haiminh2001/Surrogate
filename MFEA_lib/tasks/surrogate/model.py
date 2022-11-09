import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch_geometric.nn as gnn_nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GATConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = gnn_nn.GATConv(in_channels = in_channels, out_channels = out_channels)
        self.conv2 = gnn_nn.GATConv(in_channels = out_channels, out_channels = out_channels)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, vertices_feature, edge_index, edge_attr):
        out = self.relu(self.conv1(vertices_feature, edge_index, edge_attr))
        out = self.relu(self.conv2(out, edge_index, edge_attr))
        return out

class SurrogateModel(nn.Module):
    def __init__(self, in_channels, hid_channels, reg_max = 10000):
        super().__init__()
        hid_channels2 = hid_channels//4
        self.gcb1 = GATConvBlock(in_channels=in_channels, out_channels=hid_channels)
        self.gcb2 = GATConvBlock(in_channels=hid_channels, out_channels=hid_channels*2)
        self.gcb3 = GATConvBlock(in_channels=hid_channels*2, out_channels=hid_channels2)
        
        self.linear = nn.Linear(10, 1)
        self.regress = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hid_channels2,1)
            )
        self.reg_max = reg_max
        # self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        # self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
        #                                            requires_grad=False)
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hid_channels2,1),
            nn.Sigmoid(),
        )
        
    def forward(self, inputs):
        vertices_feature, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        x = self.gcb1(vertices_feature, edge_index, edge_attr)
        x = self.gcb2(x, edge_index, edge_attr)
        x = self.gcb3(x, edge_index, edge_attr)
        x = x.squeeze(1)
        x = self.linear(torch.transpose(x, 0, 1)).squeeze()
        # _, (_, x) = self.lstm(x)
        v = self.regress(x)
        c = self.classify(x)
        return v.flatten(), c.flatten()