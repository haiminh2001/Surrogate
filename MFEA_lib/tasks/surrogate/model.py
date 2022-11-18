import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch_geometric.nn as gnn_nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau
from sklearn.metrics import f1_score, confusion_matrix, r2_score
import os
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add
from torch_geometric.nn.dense.linear import Linear

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch_geometric.nn as gnn_nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau
from sklearn.metrics import f1_score, confusion_matrix, r2_score
import os
from .lambdarank import lambda_rank
from .gnn_block import MyGATConv

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class GATConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = MyGATConv(in_channels = in_channels, out_channels = out_channels, edge_dim = edge_dim, add_self_loops = False, concat = False, improve = True)
        self.conv2 = MyGATConv(in_channels = out_channels, out_channels = out_channels, edge_dim = out_channels, add_self_loops = False, concat = False, improve = True)
        self.relu = nn.ReLU(inplace = True)
        

    def forward(self, vertices_feature, edge_index, edge_attr):
        out, edge_attr = self.conv1(vertices_feature, edge_index, edge_attr.type(torch.float))
        out, edge_attr = self.conv2(out, edge_index, edge_attr.type(torch.float))
        return out, edge_attr

class SurrogateModel(nn.Module):
    def __init__(self, in_channels, hid_channels, reg_max = 10000):
        super().__init__()
        hid_channels2 = hid_channels//1
        self.gcb1 = GATConvBlock(in_channels=in_channels, out_channels=hid_channels, edge_dim = 91)
        self.gcb2 = GATConvBlock(in_channels=hid_channels, out_channels=hid_channels*2, edge_dim = hid_channels)
        self.gcb3 = GATConvBlock(in_channels=hid_channels*2, out_channels=hid_channels2, edge_dim = hid_channels * 2)
        
        self.node_linear = nn.Linear(10, 1)
        self.edge_linear = nn.Linear(1000,1)
        self.regress = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hid_channels2,1)
            )
        self.reg_max = reg_max
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hid_channels2,1),
            nn.Sigmoid(),
        )
        
    def forward(self, inputs):
        vertices_feature, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        x, edge_attr = self.gcb1(vertices_feature, edge_index, edge_attr)
        x, edge_attr = self.gcb2(x, edge_index, edge_attr)
        x, edge_attr = self.gcb3(x, edge_index, edge_attr)
        edge_attr = edge_attr.squeeze()
        x = self.node_linear(torch.transpose(x, 0, 1)).squeeze()
        edge_attr = self.edge_linear(torch.transpose(edge_attr, 0, 1)).squeeze()
        x = x + edge_attr
        v = self.regress(x)
        c = self.classify(x)
        return v.flatten(), c.flatten()

class SurrogatePipeline():
    def __init__(self, input_dim, hidden_dim, learning_rate, num_epochs = 1, device = 'cuda', epoch_start = None, backward_freq = 16):
        self.device = device
        self.model = SurrogateModel(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reg_criteria = nn.MSELoss()
        self.cls_criteria = nn.BCELoss()
        self.num_epochs = num_epochs
        self.eval_frequency = 10
        self.save_frequency = 1
        self.log_folder = "./checkpoints/model"
        self.load_weights_path = None
        self.epoch = 0 if epoch_start is None else epoch_start
        self.thresh_hold_cls = 0.6
        self.backward_freq = backward_freq
        self.regress_loss_weight = 10

    def train(self, train_dataset, valid_dataset):
        print("Training surrogate")
        print(self.model)
        print(f"Number of params: {self.number_of_parameters()}")
        print(f"Length of train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)}")
        print(f"Using {self.device}")
        
        self.save_opts()
        if self.load_weights_path is not None:
          self.load_model()


        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory= True, num_workers= 2)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory= True, num_workers= 2)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        y_pred_prev = None
        y_prev = None
        for self.epoch in range(self.epoch, self.num_epochs + self.epoch):
            self.model.train()

            losses_all = []
            losses_reg = []
            losses_cls = []
            losses_lambda = []
            vpreds_all = []
            vgts_all = []
            cpreds_all = []
            cgts_all = []

            skill_factor_all = []
            loss = 0 
            for i, batch in enumerate(train_dataloader):
                vpreds, cpreds = self.model(batch.to(self.device))
                logits_preds = (cpreds >= self.thresh_hold_cls).float()
                vpreds_all += list(vpreds.detach().cpu().numpy())
                vgts_all += list(batch.y.detach().cpu().numpy())
                cpreds_all += list(cpreds.detach().cpu().numpy())
                cgts_all += list(batch.thresh_hold.detach().cpu().numpy())
                skill_factor_all += list(batch.skill_factor.detach().cpu().numpy())

                #reg
                alpha = (logits_preds + batch.thresh_hold < 2).float()
                # loss_reg = self.reg_criteria(vpreds / batch.y, torch.tensor([1], device = self.device, dtype = torch.float)) 
                loss_reg = self.reg_criteria(vpreds, batch.y) 
                # print('hello', vpreds, batch.y, alpha)

                #cls
                loss_cls = self.cls_criteria(cpreds, batch.thresh_hold) * 0

                #lambda rank
                if y_prev:
                  loss_lambda = lambda_rank(y_prev, batch.y, y_pred_prev, vpreds) * 0
                  losses_lambda.append(loss_lambda.item())

                  # print(loss_lambda, loss)
                  loss += loss_lambda


                y_prev, y_pred_prev = batch.y, vpreds

                loss+= self.regress_loss_weight*loss_reg + loss_cls 

               
                losses_reg.append(loss_reg.item())
                losses_cls.append(loss_cls.item())
                losses_all.append(loss.item())

                if (i + 1) % self.backward_freq == 0:
                  loss /= self.backward_freq
                  self.optimizer.zero_grad()
                  loss.backward()
                  self.optimizer.step()
                  loss = 0
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric = {}
            vpreds_all = np.array(vpreds_all)
            vgts_all = np.array(vgts_all)
            cpreds_all = np.array(cpreds_all)
            cgts_all = np.array(cgts_all)
            skill_factor_all = np.array(skill_factor_all)

            skill_factor_value = np.unique(skill_factor_all)
            cpreds_all = (np.array(cpreds_all) >= self.thresh_hold_cls).astype(float)

            for skf in skill_factor_value:
                metric['mean'] = vpreds_all.mean()
                metric['std'] = vpreds_all.std()
                metric[f'kendalltau_{skf}'] = kendalltau(vpreds_all[skill_factor_all == skf], vgts_all[skill_factor_all == skf])
                metric[f'r2_score_{skf}'] = r2_score(vpreds_all[skill_factor_all == skf], vgts_all[skill_factor_all == skf])
                metric[f'f1_score_{skf}'] = f1_score(cpreds_all[skill_factor_all == skf], cgts_all[skill_factor_all == skf])
            
            losses = dict()
            losses['loss'] = np.mean(losses_all)
            losses['loss_cls'] = np.mean(losses_cls)
            losses['loss_reg'] = np.mean(losses_reg)
            losses['loss_lambda'] = np.mean(losses_lambda)

            self.log_terminal("Train", losses, metric)
            self.log_file("Train", losses, metric)

            if (self.epoch + 1) % self.eval_frequency == 0:
                self.eval()
            if (self.epoch + 1) % self.save_frequency == 0:
                self.save_model()
    def eval(self):
        self.model.eval()
        
        losses_all = []
        losses_cls = []
        losses_reg = []

        vpreds_all = []
        vgts_all = []
        cpreds_all = []
        cgts_all = []
        skill_factor_all = []

        with torch.no_grad():
            for _, batch in enumerate(self.valid_dataloader):
                vpreds, cpreds = self.model(batch.to(self.device))
                logits_preds = (cpreds >= self.thresh_hold_cls).float()
                vpreds_all += list(vpreds.detach().cpu().numpy())
                vgts_all += list(batch.y.detach().cpu().numpy())
                cpreds_all += list(cpreds.detach().cpu().numpy())
                cgts_all += list(batch.thresh_hold.detach().cpu().numpy())
                skill_factor_all += list(batch.skill_factor.detach().cpu().numpy())

                # loss_reg = self.reg_criteria(vpreds / batch.y, torch.Tensor([1]).type(torch.float).cuda())
                alpha = (logits_preds + batch.thresh_hold < 2).float()
                loss_reg = (1-alpha)*self.reg_criteria(vpreds, batch.y)
                loss_cls = self.cls_criteria(cpreds, batch.thresh_hold)
                loss = self.regress_loss_weight*loss_reg + loss_cls
                losses_reg.append(loss_reg.item())
                losses_cls.append(loss_cls.item())
                losses_all.append(loss.item())
 
        metric = {}
        vpreds_all = np.array(vpreds_all)
        vgts_all = np.array(vgts_all)
        cpreds_all = np.array(cpreds_all)
        cgts_all = np.array(cgts_all)
        skill_factor_all = np.array(skill_factor_all)

        skill_factor_value = np.unique(skill_factor_all)
        cpreds_all = (np.array(cpreds_all) >= self.thresh_hold_cls).astype(float)

        for skf in skill_factor_value:
            metric[f'kendalltau_{skf}'] = kendalltau(vpreds_all[skill_factor_all == skf], vgts_all[skill_factor_all == skf])
            metric[f'r2_score_{skf}'] = r2_score(vpreds_all[skill_factor_all == skf], vgts_all[skill_factor_all == skf])
            metric[f'f1_score_{skf}'] = f1_score(cpreds_all[skill_factor_all == skf], cgts_all[skill_factor_all == skf])
        
        losses = dict()
        losses['loss'] = np.mean(losses_all)
        losses['loss_cls'] = np.mean(losses_cls)
        losses['loss_reg'] = np.mean(losses_reg)

        self.log_terminal("Validation", losses, metric)
        self.log_file("Validation", losses, metric)

    def predict(self, input):
        with torch.no_grad():
            input.to(self.device)
            pred = self.model(input)
            return pred

    def number_of_parameters(self):
        return count_parameters(self.model)

    def log_terminal(self, mode, loss, metric):
        print_string = "epoch {} [{}] " + " | loss: {} + | metric: {}"
        print(print_string.format(self.epoch, mode, loss, metric))

    def log_file(self, mode, loss, metric):
        os.makedirs(self.log_folder, exist_ok=True)
        print_string = "epoch {} [{}] " + " | loss: {} | metric: {} \n"
        with open(f"{self.log_folder}/log.txt", "a") as f:
            f.write(print_string.format(self.epoch, mode, loss, metric))

    def save_opts(self):
        # os.makedirs(self.log_folder, exist_ok=True)
        # opt = self.__class__.__dict__.copy()
        # print(opt)
        # opt.pop("model")
        # opt.pop("optimizer")
        # opt.pop("reg_criteria")
        # opt.pop("cls_criteria")
        # with open(os.path.join(self.log_folder, 'opt.json'), 'w') as f:
        #     json.dump(opt, f, indent=2) 
        pass
        

    def save_model(self):
        print("Save model to:", self.log_folder)
        os.makedirs(self.log_folder, exist_ok=True)
        log_path = os.path.join(self.log_folder, "weights_{}.pth".format(self.epoch))
        torch.save(self.model.state_dict(), log_path) 

    def load_model(self):
        print("Load model from:", self.load_weights_path)
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.load_weights_path)
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load(model_dict)

    def save_pipeline(self, path):
        pass
