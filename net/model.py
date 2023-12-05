# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from typing import List


class GNNModel(nn.Module):
    def __init__(self, input_dim, hiddenunits: List[int], num_classes, prob_matrix, bias=True, drop_prob=0.5):
        super(GNNModel, self).__init__()

        self.input_dim = input_dim
        
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()
        # requires_grad=False 不参与训练
        self.prob_matrix = nn.Parameter((torch.FloatTensor(prob_matrix)), requires_grad=False)
        
        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i-1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)
        
        self.act_fn = nn.ReLU()

    def forward(self, seed_vec):
        
        for i in range(self.input_dim - 1):
            if i == 0:
                mat = self.prob_matrix.T @ seed_vec.T
                attr_mat = torch.cat((seed_vec.T.unsqueeze(0), mat.unsqueeze(0)), 0)
            else:
                mat = self.prob_matrix.T @ attr_mat[-1]
                attr_mat = torch.cat((attr_mat, mat.unsqueeze(0)), 0)
        
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_mat.T)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = torch.sigmoid(self.fcs[-1](self.dropout(layer_inner)))
        return res
    
    def loss(self, y, y_hat):
        forward_loss = F.mse_loss(y_hat, y)
        return forward_loss


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        
        self.bn = nn.BatchNorm1d(num_features=latent_dim)
        
    def forward(self, x):
        h_ = F.relu(self.FC_input(x))
        h_ = F.relu(self.FC_input2(h_))
        h_ = F.relu(self.FC_input2(h_))
        # 算p(Z|X)的均值和方差
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        #self.prelu = nn.PReLU()
        
    def forward(self, x):
        h = F.relu(self.FC_input(x))
        h = F.relu(self.FC_hidden_1(h))
        h = F.relu(self.FC_hidden_2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        std = torch.exp(0.5*var) # standard deviation
        epsilon = torch.randn_like(var)
        return mean + std*epsilon

    def forward(self, x, adj=None):
        if adj != None:
            mean, log_var = self.Encoder(x, adj)
        else:
            mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, log_var) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var


class DiffusionPropagate(nn.Module):
    def __init__(self, prob_matrix, niter):
        super(DiffusionPropagate, self).__init__()
        
        self.niter = niter 
        
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()
        
        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))
    
    def forward(self, preds, seed_idx):
        # import ipdb; ipdb.set_trace()
        # prop_preds = torch.ones((preds.shape[0], preds.shape[1])).to(device)
        device = preds.device
        
        for i in range(preds.shape[0]):
            prop_pred = preds[i]
            for j in range(self.niter):
                P2 = self.prob_matrix.T * prop_pred.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                prop_pred = torch.ones((self.prob_matrix.shape[0], )).to(device) - torch.prod(P3, dim=1)
                # prop_pred[seed_idx[seed_idx[:,0] == i][:, 1]] = 1
                prop_pred = prop_pred.unsqueeze(0)
            if i == 0:
                prop_preds = prop_pred
            else:
                prop_preds = torch.cat((prop_preds, prop_pred), 0)
                
        return prop_preds