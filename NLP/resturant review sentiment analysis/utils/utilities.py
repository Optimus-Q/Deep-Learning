#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Import libraries..

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def TestAccuracy(pred, yt):
    
    pred_ix = pred.argmax(dim = 1)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(yt.numpy(), pred_ix.numpy())
    return (np.round(acc, 3))

def ValidationAnalysis(model, criterion, valid_loader):
    valid_actual = []
    valid_predict = []
    avg_valid_loss = []
    with torch.no_grad():
        
        for xv, yv in valid_loader:
            yp = model(xv)
            pred_ix = yp.argmax(dim = 1)
            valid_loss = criterion(yp, yv)
            avg_valid_loss.append(valid_loss)
            valid_predict.append(pred_ix.numpy())
            valid_actual.append(yv.numpy())
                     
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(np.array(valid_actual).reshape(-1,1), np.array(valid_predict).reshape(-1,1)) 
    return (yp, np.array(valid_predict).reshape(-1,1), np.array(valid_actual).reshape(-1,1), np.round(acc, 3), np.round(np.mean(avg_valid_loss),3))

def EachAccuracyClass(yt, pt):
    cls, ncls = np.unique(yt, return_counts = True)
    acc = pd.DataFrame()
    acc['ytest'] = yt.squeeze()
    acc['pred'] = pt.squeeze()
    class_acc = {}
    from sklearn.metrics import accuracy_score
    for c in cls:
        sig = acc.loc[acc.loc[acc['ytest']==c].index]
        class_acc[str(c)] = round(accuracy_score(sig['ytest'], sig['pred'])*100, 2)
    return (class_acc)

