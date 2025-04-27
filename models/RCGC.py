import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
import logging
import math
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F

#GraFormer

"""
@inproceedings{wu2021graph,
title={Graph-based 3d multi-person pose estimation using multi-view images},
author={Wu, Size and Jin, Sheng and Liu, Wentao and Bai, Lei and Qian, Chen and Liu, Dong and Ouyang, Wanli},
booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
pages={11148--11157},
year={2021}
}
"""

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

class ChebnetII_prop(MessagePassing):
    def __init__(self, K, Init=False, bias=True, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.Init=Init
        self.reset_parameters()
        self.node_dim = 1
    def reset_parameters(self):
        self.temp.data.fill_(1.0)

        if self.Init:
            for j in range(self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                self.temp.data[j] = x_j**2
        
    def forward(self, x, edge_index,edge_weight=None):
        coe_tmp=F.relu(self.temp)
        coe=coe_tmp.clone()
        
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)
        
        edge_index = edge_index.to(torch.long)
        
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))
        # x = x.to('cuda')
        edge_index_tilde = edge_index_tilde.to('cuda')
        norm_tilde = norm_tilde.to('cuda')
        Tx_0=x
        Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)

        out=coe[0]/2*Tx_0+coe[1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
            Tx_2=2*Tx_2-Tx_0
            out=out+coe[i]*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

def adj_mx_from_spectral_similarity(features):

    features = features.detach().cpu().numpy()

    if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features) 
    adj = cosine_similarity(features)
    np.fill_diagonal(adj, 1)
    adj = torch.tensor(adj, dtype=torch.float32)
    
    return adj

def extract_edges_from_adj(adj_matrix, threshold=0.5):
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    
    adj_matrix = torch.triu(adj_matrix, diagonal=1)
    
    row, col = torch.where(adj_matrix > threshold)
    
    edges = torch.stack((row, col), dim=0)
    
    return edges

class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = ChebnetII_prop(K=2, Init=True)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, adj):
        x = self.gconv(x, adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        else:
            pass
        return x


class RCGC(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_seq, p_dropout):
        super(RCGC, self).__init__()
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)

    def forward(self, x):
        
        features = x.mean(dim=0)  
        
        adj = adj_mx_from_spectral_similarity(features)
        adj = extract_edges_from_adj(adj, threshold=0.95)

        out = self.gconv1(x, adj)
        return out

