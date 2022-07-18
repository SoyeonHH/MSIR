from sched import scheduler
import torch
from torch import float32, nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import *
from encoder import *
import pickle

def sent2intensity(self, labels):
    intensity = []
    for label in labels:
        if label < -15/7 or label > 15/7:
            intensity.append(3)
        elif label < -9/7 or label > 9/7:
            intensity.append(2)
        elif label < -3/7 or label > 3/7:
            intensity.append(1)
        else:
            intensity.append(0)
    return intensity

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

class Intensity(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model
        
        self.model_name = model_name = hp.model_name

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Pre-encoding per unimodal for frozen model architecture
        self.text_emb = LanguageEmbeddingLayer(hp)
        self.text_enc = TextSubNet(hp.d_tin, hp.d_th, hp.d_tout, dropout=0.15)
        self.audio_enc = SubNet(hp.d_ain, hp.d_ah, hp.dropout_a)
        self.video_enc = SubNet(hp.d_vin, hp.d_vh, hp.dropout_v)

        self.text_emb, self.text_enc, self.audio_enc, self.video_enc = \
            self.text_emb.to(self.device), self.text_enc.to(self.device), self.audio_enc.to(self.device), self.video_enc.to(self.device)

        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta

        self.update_batch = hp.update_batch

        # initialize the model
        if model_name == 'TFN':
            self.model = model = TFN(hp)
        elif model_name == 'MIM':
            self.model = model = MMIM(hp)
        elif model_name == 'MAG':
            multimodal_config = MultimodalConfig(
            beta_shift=hp.beta_shift, dropout_prob=hp.dropout_prob
            )  
            self.model = model = MAG_BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', multimodal_config=multimodal_config, num_labels=1
            )
        else:
            print("The Configuration no exist.")

        self.model = model = self.model.to(self.device)

        # criterion - mosi and mosei are regression datasets
        self.criterion = criterion = nn.L1Loss(reduction="mean")

        # optimizer
        self.optimizer={}

        if self.is_train:
            main_param = []
            bert_param = []

            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    else:
                        main_param.append(p)
                
        optimizer_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer = getattr(torch.optim, self.hp.optim)(
            optimizer_group
        )

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=hp.when, factor=0.5, verbose=True)

    
    def train_and_eval_fusion(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion

        def train(model, optimizer, criterion):
            epoch_loss = 0

            model.train()