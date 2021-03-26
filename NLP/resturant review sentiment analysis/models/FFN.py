#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForwardSentiment(nn.Module):
    
    def __init__(self, num_embed, embed_dim, batch, ip1, op1, ip2, op2, ip3, op3, ip4, op4, ip5, op5):
        super(FeedForwardSentiment, self).__init__()
        
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.batch = batch
        self.ip1 = ip1
        self.op1 = op1
        self.ip2 = ip2
        self.op2 = op2
        self.ip3 = ip3
        self.op3 = op3
        self.ip4 = ip4
        self.op4 = op4
        self.ip5 = ip5
        self.op5 = op5
        
        self.embed = nn.Embedding(self.num_embed, self.embed_dim)
        self.fc1 = nn.Linear(self.ip1, self.op1)
        self.fc2 = nn.Linear(self.ip2, self.op2)
        self.fc3 = nn.Linear(self.ip3, self.op3)
        self.fc4 = nn.Linear(self.ip4, self.op4)
        self.fc5 = nn.Linear(self.ip5, self.op5)
        self.drop = nn.Dropout(0.3)
        self.prob = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.view(self.batch, self.ip1)
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.drop(x)
        x = F.tanh(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.prob(x)
        return(x)    

