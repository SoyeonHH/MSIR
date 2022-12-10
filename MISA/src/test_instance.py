from cgi import test
import os
from pyexpat import model
import sys
import math
from math import isnan
import re
import pickle
import gensim
from create_dataset import PAD
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from transformers import BertTokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import config

from utils import to_gpu, to_cpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
from utils.tools import *
import models

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def sent2class(test_preds_sent):
    preds_2, preds_7 = [], []

    for pred in test_preds_sent:
        # preds_2 appending
        # print(pred)
        pred = pred.any()
        if pred > 0:
            preds_2.append('pos')
        else:
            preds_2.append('neg')

        # preds_7 appending
        if pred < -15/7:
            preds_7.append('very negative')
        elif pred < -9/7:
            preds_7.append('negative')
        elif pred < -3/7:
            preds_7.append('weakly negative')
        elif pred < 3/7:
            preds_7.append('Neutral')
        elif pred < 9/7:
            preds_7.append('weakly positive')
        elif pred < 15/7:
            preds_7.append('positive')
        else:
            preds_7.append('very positive')

    assert len(test_preds_sent) == len(preds_2) == len(preds_7)
    return preds_2, preds_7


class TestMOSI(object):
    def __init__(self, solver, test_config, test_data_loader):
        self.model = solver.model
        self.config = test_config
        self.test_loader = test_data_loader
        self.H = []
        self.config.use_confidNet = False

    def start(self):
        segment_list, labels, labels_2, labels_7, preds, preds_2, preds_7 = \
            [], [], [], [], [], [], []

        model = self.model
        
        model.load_state_dict(torch.load(f"pre_trained_models/best_model_MISA_{self.config.data}.pt"))
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):

                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                # Gold-truth
                labels.extend(y)
                
                segment_list.extend(ids)
                
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                # l = to_gpu(l)
                l = to_cpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                # Predictions
                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                y_tilde = y_tilde.squeeze()

                preds.extend(y_tilde.cpu().detach().numpy())
                # self.H.extend(H)

            labels_2, labels_7 = sent2class(labels)
            preds_2, preds_7 = sent2class(preds)
            
        test_dict = \
            {'segment': segment_list,
            'labels': labels,
            'labels_2': labels_2,
            'labels_7': labels_7,
            'preds': preds,
            'preds_2': preds_2,
            'preds_7': preds_7,
            }
        
        path = os.getcwd() + '/results/' + 'MISA_' + self.config.data + '_confidNet.pkl'
        to_pickle(test_dict, path)
        # save_hidden(self.H, self.config.data)