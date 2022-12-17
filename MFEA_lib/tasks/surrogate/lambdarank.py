import torch

from torch.nn.functional import binary_cross_entropy_with_logits
def lambda_rank(y1, y2, y_pred1, y_pred2):
  if y1 == y2:
    return torch.tensor(0, device = 'cuda:0', dtype= torch.float)

  s_ij = torch.sigmoid(y1 - y2)
  s_ij = torch.tensor([s_ij], dtype = torch.float, device = 'cuda:0')
  p_hat_ij = torch.sigmoid(y_pred1 - y_pred2) 
  
  return binary_cross_entropy_with_logits(p_hat_ij, s_ij)