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

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model = None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model
        
        self.model_name = model_name = hp.model_name
        self.U = []
        self.H = []

        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")

        # # Pre-encoding per unimodal for frozen model architecture
        # self.text_emb = LanguageEmbeddingLayer(hp)
        # self.text_enc = TextSubNet(hp.d_tin, hp.d_th, hp.d_tout, dropout=0.15)
        # self.audio_enc = SubNet(hp.d_ain, hp.d_ah, hp.dropout_a)
        # self.video_enc = SubNet(hp.d_vin, hp.d_vh, hp.dropout_v)

        # self.text_emb, self.text_enc, self.audio_enc, self.video_enc = \
        #     self.text_emb.to(self.device), self.text_enc.to(self.device), self.audio_enc.to(self.device), self.video_enc.to(self.device)

        self.update_batch = hp.update_batch

        # initialize the model
        if model_name == 'TFN':
            self.model = model = TFN(hp)
        elif model_name == 'Glove':
            self.model = model = Text(hp)
        elif model_name == 'Facet':
            self.model = model = Visual(hp)
        elif model_name == 'COVAREP':
            self.model = model = Acoustic(hp)

        self.model = model = self.model.to(self.device)

        # criterion - mosi and mosei are regression datasets
        self.criterion = nn.L1Loss(reduction="mean")

        # optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=hp.when, factor=0.5, verbose=True)


    # trianing and evalution
    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        # creterion for downstream task
        criterion = self.criterion

        def train(model, optimizer, criterion):
            epoch_loss = 0

            model.train()
            model.zero_grad()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            start_time = time.time()

            left_batch = self.update_batch

            for i_batch, batch_data in enumerate(tqdm(self.train_loader)):
                
                text, visual, vlens, audio, alens, y, l, glove_sent, bert_sent, bert_sent_type, \
                    bert_sent_mask, ids = batch_data

                text, visual, audio, y, glove_sent = \
                    text.to(self.device), visual.to(self.device), audio.to(self.device), y.to(self.device), glove_sent.to(self.device)

                device = torch.device('cpu')
                vlens, alens, l = vlens.to(device), alens.to(device), l.to(device)

                batch_size = y.size(0)
                
                if self.hp.model_name == 'TFN':
                    preds, H = model(audio, visual, glove_sent, alens, vlens, l)
                elif self.hp.model_name == 'Glove':
                    preds, H = model(glove_sent, l)
                elif self.hp.model_name == 'Facet':
                    preds, H = model(visual, vlens)
                elif self.hp.model_name == 'COVAREP':
                    preds, H = model(audio, alens)
                
                loss = criterion(preds, y)
                loss.backward()

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer.step()

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                
                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, avg_loss))
                    proc_loss, proc_size = 0, 0
                    start_time = time.time()

            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            
            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, visual, vlens, audio, alens, y, lengths, glove_sent, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    text, visual, audio, y, glove_sent = \
                        text.to(self.device), visual.to(self.device), audio.to(self.device), y.to(self.device), glove_sent.to(self.device)

                    device = torch.device('cpu')
                    vlens, alens, l = vlens.to(device), alens.to(device), lengths.to(device)
                    
                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)
                    
                    if self.hp.model_name == 'TFN':
                        preds, H = model(audio, visual, glove_sent, alens, vlens, l)
                    elif self.hp.model_name == 'Glove':
                        preds, H = model(glove_sent, l)
                    elif self.hp.model_name == 'Facet':
                        preds, H = model(visual, vlens)
                    elif self.hp.model_name == 'COVAREP':
                        preds, H = model(audio, alens)
                    # self.H.extend(H)
                    
                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        best_valid = 1e8
        best_mae = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            train_loss = train(model, optimizer, criterion)

            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)

            end = time.time()
            duration = end-start
            scheduler.step(val_loss)    # Decay learning rate by validation loss

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss

                if test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    if self.hp.dataset in ["mosei_senti", "mosei"]:
                        eval_mosei_senti(results, truths, True)

                    elif self.hp.dataset == 'mosi':
                        eval_mosi(results, truths, True)

                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models/MM.pt!")
                    save_model(model, self.model_name, self.hp.dataset)
            else:
                patience -= 1
                if patience == 0:
                    break

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)

        # save_hidden(self.H, self.modality)
        # save_hidden(self.H_out, self.modality + '_out')
        # save_hidden(self.H, self.model_name, self.hp.dataset)
        sys.stdout.flush()