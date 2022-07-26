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
        self.model_name = hp.model_name
        self.U = []
        self.H = []

        model = TFN((hp.d_ain, hp.d_vin, hp.d_tin), (32, 32, 128), 128, (0.15, 0.15, 0.15, 0.15), 128)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model = model.to(self.device)
        print("Model initialized")
        self.criterion = nn.L1Loss(size_average=False)
        self.optimizer = optim.Adam(list(model.parameters())[2:]) # don't optimize the first 2 params, they should be fixed (output_range and shift)

    # trianing and evalution
    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer

        # creterion for downstream task
        criterion = self.criterion

        def train(model, optimizer, criterion):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            start_time = time.time()

            for i_batch, batch_data in enumerate(tqdm(self.train_loader)):
                
                text, visual, audio, y, _ = batch_data

                device = self.device
                text, visual, audio, y = \
                    text.to(device), visual.to(device), audio.to(device), y.to(device)

                batch_size = y.size(0)

                '''
                audio: tensor of shape (batch_size, audio_in)
                visual: tensor of shape (batch_size, video_in)
                text_emb: tensor of shape (batch_size, sequence_len, text_in)'''
                audio = torch.Tensor.mean(audio, dim=0, keepdim=True)
                visual = torch.Tensor.mean(visual, dim=0, keepdim=True)
                audio = audio[0,:,:]
                visual = visual[0,:,:]
                
                preds, H = model(audio, visual, text)
                loss = criterion(preds, y)
                loss.backward()

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                optimizer.step()
                
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
                    text, visual, audio, y, _ = batch

                    # with torch.cuda.device(0):
                    device = self.device
                    text, visual, audio, y = \
                        text.to(device), visual.to(device), audio.to(device), y.to(device)
                    
                    batch_size = y.size(0) # bert_sent in size (bs, seq_len, emb_size)
                    audio = torch.Tensor.mean(audio, dim=0, keepdim=True)
                    visual = torch.Tensor.mean(visual, dim=0, keepdim=True)
                    audio = audio[0,:,:]
                    visual = visual[0,:,:]

                    preds, H = model(audio, visual, text)
                    self.H.extend(H)

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
        save_hidden(self.H, self.model_name, self.hp.dataset)
        sys.stdout.flush()