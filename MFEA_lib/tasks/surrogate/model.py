import torch.nn as nn
import torch 
import torch.nn.functional as F
import torch_geometric.nn as gnn_nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

  
# type 2
class GNN_GCN(nn.Module):
    def __init__(self, in_channels, hid_channels, num_nodes):
        super(GNN_GCN, self).__init__()
        self.gc1 = gnn_nn.GCNConv(in_channels, hid_channels)
        self.gc2 = gnn_nn.GCNConv(hid_channels, 1)
        self.fc = nn.Linear(num_nodes, 1)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.gc1(x, edge_index, edge_weight))
        x = self.gc2(x, edge_index, edge_weight)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

# type 1
class SurrogateModel(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.gc1 = gnn_nn.GATConv(in_channels=in_channels, out_channels=hid_channels)
        self.gc2 = gnn_nn.GATConv(in_channels=hid_channels, out_channels=256)
        self.lstm = nn.LSTM(256, 256)
        self.regress = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256,1)
            )
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
        
    def forward(self, inputs):
        vertices_feature, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        x = F.relu(self.gc1(vertices_feature, edge_index, edge_attr))
        x = self.gc2(x, edge_index, edge_attr)
        x = x.squeeze(1)
        _, (_, x) = self.lstm(x)
        v = self.regress(x)
        c = self.classify(x)
        return v.flatten(), c.flatten()
    
class SurrogatePipeline():
    def __init__(self, input_dim, hidden_dim, learning_rate, num_epochs = 1, device = 'cuda'):
        self.device = device
        self.model = SurrogateModel(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reg_criteria = nn.MSELoss()
        self.cls_criteria = nn.BCELoss()
        self.num_epochs = num_epochs
        self.eval_frequency = 2
        self.save_frequency = 1
        self.log_folder = "./checkpoints/model"
        self.load_weights_path = None

    def train(self, train_dataset, valid_dataset):
        print("Training surrogate")
        print(self.model)
        print(f"Number of params: {self.number_of_parameters()}")
        print(f"Length of train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)}")
        print(f"Using {self.device}")
        if self.load_weights_path is not None:
          self.load_model()


        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory= True, num_workers= 2)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory= True, num_workers= 2)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epoch = 0
        for self.epoch in range(self.num_epochs):
            self.model.train()

            losses = []
            vpreds_all = []
            vgts_all = []
            cpreds_all = []
            cgts_all = []
            
            for _, batch in tqdm(enumerate(train_dataloader)):
                vpreds, cpreds = self.model(batch.to(self.device))
                vpreds_all += list(vpreds.detach().cpu().numpy())
                vgts_all += list(batch.y.detach().cpu().numpy())
                # cpreds_all += list(cpreds.squeeze(-1).numpy())
                # cgts_all += list(batch.y.squeeze(-1).numpy())
                loss = self.reg_criteria(vpreds / batch.y, torch.Tensor([1]).type(torch.float).cuda()) + self.cls_criteria(cpreds, batch.thresh_hold)
                losses.append(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.log_terminal("Train", np.mean(losses), kendalltau(vpreds_all, vgts_all))

            if (self.epoch + 1) % self.eval_frequency == 0:
                self.eval()
            if (self.epoch + 1) % self.save_frequency == 0:
                self.save_model()

    def eval(self):
        self.model.eval()
        losses = []
        vpreds_all = []
        vgts_all = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.valid_dataloader)):
                vpreds, cpreds = self.model(batch.to(self.device))
                vpreds_all += list(vpreds.detach().cpu().numpy())
                vgts_all += list(batch.y.detach().cpu().numpy())
                loss = self.reg_criteria(vpreds / batch.y, torch.Tensor([1]).type(torch.float).cuda()) + self.cls_criteria(cpreds, batch.thresh_hold)
                losses.append(loss.item())

        self.log_terminal("Validation", np.mean(losses), kendalltau(vpreds_all, vgts_all))

    def predict(self, input):
        with torch.no_grad():
            input.to(self.device)
            pred = self.model(input)
            return pred

    def number_of_parameters(self):
        return count_parameters(self.model)

    def log_terminal(self, mode, loss, metric):
        print_string = "epoch {:>3} [{}] " + " | loss: {:.5f} | metric: {}"
        print(print_string.format(self.epoch, mode, loss, metric))

    def save_opts(self):
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
