from cgi import test
import os
from pyexpat import model
import sys
import math
from math import isnan
import re
import pickle
from data_loader import get_loader
import gensim
from create_dataset import PAD
from config import *
from utils.tools import *
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
    def __init__(self, hp, solver):
        self.hp = hp
        self.H = []

        dataset = str.lower(hp.dataset.strip())

        test_config = get_config(dataset, mode='test',  batch_size=hp.batch_size)
        self.test_loader = get_loader(hp, test_config, shuffle=False)
        self.model = solver.model

    def start(self):
        segment_list, labels, labels_2, labels_7, preds, preds_2, preds_7 = \
            [], [], [], [], [], [], []

        model = self.model
        
        model.load_state_dict(torch.load(f"pre_trained_models/best_model_MIM_{self.hp.dataset}.pt"))
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):

                text, visual, vlens, audio, alens, y, lengths, \
                    bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                
                segment_list.extend(ids)
                
                # Gold-truth
                labels.extend(y)
                
                # Predictions
                device = torch.device('cuda')
                text, audio, visual, y = text.to(device), audio.to(device), visual.to(device), y.to(device)
                lengths = lengths.to(device)
                bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
                
                _, _, logits, _, _, H = model(text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                preds.extend(logits.cpu().detach().numpy())
                self.H.extend(H)

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
        

        path = os.getcwd() + '/MMIM/dist/'+ 'MMIM_' + self.hp.dataset + '.pkl'
        to_pickle(test_dict, path)

        path = os.getcwd() + '/results/' + 'MIM_' + self.hp.dataset + '.pkl'
        to_pickle(test_dict, path)
        save_hidden(self.H, self.hp.dataset)
