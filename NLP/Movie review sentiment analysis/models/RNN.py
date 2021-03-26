#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn import functional as F

class RnnSentiment(nn.Module):
    
    def __init__(self, num_embed, embed_dim, inp_sz, rnn_hidsz, nlayer, seq_len, batch, fc_hidsz, out_sz):
        super(RnnSentiment, self).__init__()
        
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.inp_sz = inp_sz
        self.rnn_hidsz = rnn_hidsz
        self.nlayer = nlayer
        self.seq_len = seq_len
        self.batch = batch
        self.fc_hidsz  = fc_hidsz
        self.out_sz = out_sz
        
        self.embed = nn.Embedding(self.num_embed, self.embed_dim)
        self.rnn = nn.RNN(self.inp_sz, self.rnn_hidsz, self.nlayer)
        self.fc = nn.Linear(self.fc_hidsz, self.out_sz)
        self.prob = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.view(self.seq_len, self.batch, self.inp_sz)
        hidden = torch.zeros(self.nlayer, self.batch, self.rnn_hidsz)
        x, hidden = self.rnn(x, hidden)
        x = x.view(self.batch, self.fc_hidsz)
        x = self.fc(x)
        x = self.prob(x)
        return (x)

