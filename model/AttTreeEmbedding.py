import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import numpy as np 
from torch_multi_head_attention import MultiHeadAttention

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


class FusionAttention(nn.Module):
    def __init__(self,dim):
        super(FusionAttention, self).__init__()
        self.attention_matrix = nn.Linear(dim, dim)
        self.project_weight = nn.Linear(dim,1)
    def forward(self, inputs):
        query_project = self.attention_matrix(inputs) # (b,t,d) -> (b,t,d2)
        query_project = F.leaky_relu(query_project)
        project_value = self.project_weight(query_project) # (b,t,h) -> (b,t,1)
        attention_weight = torch.softmax(project_value, dim=1) # Normalize and calculate weights (b,t,1)
        attention_vec = inputs * attention_weight
        attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_weight


class Attention(nn.Module):
    def __init__(self,dim,hidden,aggregate="sum"):
        super(Attention, self).__init__()
        self.attention_matrix = nn.Linear(dim, hidden)
        self.project_weight = nn.Linear(hidden*2,hidden)
        self.h = nn.Parameter(torch.rand(hidden,1))
        self.agg_type = aggregate
    def forward(self, query,key): # assume key==value
        dim = query.size(-1)
        batch,time_step = key.size(0) ,key.size(1)
        
        # concate input query and key 
        query = query.view(batch,1,dim)
        query = query.expand(batch,time_step,-1)
        cat_vector = torch.cat((query,key),dim=-1)
        
        # project to single value
        project_vector = self.project_weight(cat_vector) 
        project_vector = torch.relu(project_vector)
        attention_alpha = torch.matmul(project_vector,self.h)
        attention_weight = torch.softmax(attention_alpha, dim=1) # Normalize and calculate weights (b,t,1)
        attention_vec = key * attention_weight
        
        # aggregate leaves
        if self.agg_type == "max":
            attention_vec,_ = torch.max(attention_vec,dim=1)
        elif self.agg_type =="mean":
            attention_vec = torch.mean(attention_vec,dim=1)
        elif self.agg_type =="sum":
            attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_weight