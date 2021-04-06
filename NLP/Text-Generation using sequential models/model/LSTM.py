#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
from torch.nn import functional as F
class LSTMTG(nn.Module):
    
    def __init__(self, num_embed, embed_dim, seq_len, inp_sz, rnn_hidsz, nlayer, fc_hidsz, out_sz):
        super(LSTMTG, self).__init__()
        
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.inp_sz = inp_sz
        self.rnn_hidsz = rnn_hidsz
        self.nlayer = nlayer
        self.fc_hidsz = fc_hidsz
        self.out_sz = out_sz
        
        self.embed = nn.Embedding(self.num_embed, self.embed_dim)
        self.rnn = nn.LSTM(self.inp_sz,self.rnn_hidsz, self.nlayer, bidirectional=False)
        self.fc = nn.Linear(self.fc_hidsz, self.out_sz)
        self.prob = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.4)
        
    def forward(self, x):
        xbatch = x.shape[0]
        x = self.embed(x)
        x = x.view(self.seq_len,-1,self.inp_sz)
        hidden = torch.zeros(self.nlayer, xbatch, self.rnn_hidsz)
        cell = torch.zeros(self.nlayer, xbatch, self.rnn_hidsz)
        x, (hidden, cell) = self.rnn(x, (hidden, cell))
        x = x.view(xbatch, self.fc_hidsz)
        x = self.drop(self.fc(x))
        x = self.prob(x)
        return(x)

