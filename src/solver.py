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
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            start_time = time.time()

            left_batch = self.update_batch

            for i_batch, batch_data in enumerate(tqdm(self.train_loader)):
                
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, \
                    bert_sent_mask, ids = batch_data

                device = self.device
                text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.to(device), visual.to(device), audio.to(device), y.to(device), l.to(device), bert_sent.to(device), \
                        bert_sent_type.to(device), bert_sent_mask.to(device)

                batch_size = y.size(0)

                '''
                audio: tensor of shape (batch_size, audio_in)
                visual: tensor of shape (batch_size, video_in)
                text_emb: tensor of shape (batch_size, sequence_len, text_in)'''
                audio = torch.Tensor.mean(audio, dim=0, keepdim=True)
                visual = torch.Tensor.mean(visual, dim=0, keepdim=True)
                audio = audio[0,:,:]
                visual = visual[0,:,:]
                text_emb = self.text_emb(text, bert_sent, bert_sent_type, bert_sent_mask)

                '''_h = tensor of shape (batch_size, hidden_size)'''
                text_h = self.text_enc(text_emb)
                audio_h = self.audio_enc(audio)
                video_h = self.video_enc(visual)
                
                preds, H = model(audio_h, video_h, text_h)
                
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
                    text, visual, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    # with torch.cuda.device(0):
                    device = torch.device('cuda')
                    text, audio, visual, y = text.to(device), audio.to(device), visual.to(device), y.to(device)
                    lengths = lengths.to(device)
                    bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
                    
                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)
                    audio = torch.Tensor.mean(audio, dim=0, keepdim=True)
                    visual = torch.Tensor.mean(visual, dim=0, keepdim=True)
                    audio = audio[0,:,:]
                    visual = visual[0,:,:]

                    text_emb = self.text_emb(text, bert_sent, bert_sent_type, bert_sent_mask)
                    text_h = self.text_enc(text_emb)
                    audio_h = self.audio_enc(audio)
                    video_h = self.video_enc(visual)

                    preds, H = model(audio_h, video_h, text_h)
                    self.H.extend(H)
                    
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
                    save_model(model, self.model_name + '_origin_mosei')
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
        save_hidden(self.H, self.model_name + '_origin_mosei')
        sys.stdout.flush()