import os
from pyexpat import model
import sys
import math
import re
import pickle
import gensim
from utils.tools import *
from multimodal_driver import get_appropriate_dataset, args
from global_configs import *
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
    def __init__(self, args, model):
        self.H = []

        with open(f"{DATA_DICT}/{args.dataset}.pkl", "rb") as handle:
            data = pickle.load(handle)
        
        test_data = data["test"]
        test_dataset, test_tokenizer = get_appropriate_dataset(test_data)
        self.test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

        self.model = model

    def start(self):
        segment_list, labels, labels_2, labels_7, preds, preds_2, preds_7 = \
            [], [], [], [], [], [], []

        model = self.model
        
        model.load_state_dict(torch.load(f"pre_trained_models/best_model_MAG_{args.dataset}.pt"))
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):

                batch = tuple(t.to(DEVICE) for t in batch)

                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch

                # Gold-truth
                segment_list.extend(segment_ids)
                labels.extend(label_ids)

                # Predictions
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)
                outputs = model(
                    input_ids,
                    visual,
                    acoustic,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None
                )
                logits = outputs[0]
                preds.extend(logits)
                self.H.extend(outputs)

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
        
        path = os.getcwd() + '/results/' + 'MAG_' + args.dataset + '.pkl'
        to_pickle(test_dict, path)
        save_hidden(self.H, args.dataset)