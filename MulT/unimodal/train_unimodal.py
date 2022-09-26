from email.utils import make_msgid
import torch
from torch import nn
import sys
from unimodal import models_unimodal
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
import wandb

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models_unimodal, hyp_params.model+'Model')(hyp_params)
    device = hyp_params.device

    if hyp_params.use_cuda:
        # model = model.cuda()
        model = model.to(device)

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    optimizer.param_groups[0]['capturable'] = True
    criterion = getattr(nn, hyp_params.criterion)()
    if hyp_params.aligned or hyp_params.model=='MULT':
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            # ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.to(device), ctc_v2l_module.to(device)
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    # ctc_a2l_module = settings['ctc_a2l_module']
    # ctc_v2l_module = settings['ctc_v2l_module']
    # ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    # ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    # ctc_criterion = settings['ctc_criterion']
    
    scheduler = settings['scheduler']
    device = hyp_params.device
    

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    # text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    text, audio, vision, eval_attr = text.to(device), audio.to(device), vision.to(device), eval_attr.to(device)
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
                
            loss = 0
            # net = nn.DataParallel(model) if batch_size > 10 else model
            net = model
            if batch_chunk > 1:
                loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                    
                    if hyp_params.dataset == 'iemocap':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)
                    loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    loss_i.backward()
                loss += loss_i
            else:
                preds, hiddens = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                loss = criterion(preds, eval_attr)
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        # text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        text, audio, vision, eval_attr = text.to(device), audio.to(device), vision.to(device), eval_attr.to(device)
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                # net = nn.DataParallel(model) if batch_size > 10 else model
                net = model
                preds, _ = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size
                
                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)


        results = torch.cat(results)
        truths = torch.cat(truths)
        # print()
        # print("=======================================================================================================")
        # print(results.shape)
        # print(truths.shape)
        # print("=======================================================================================================")
        # print()
       
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, results, truths = evaluate(model, criterion, test=True)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name='Unimodal'+"_"+hyp_params.modality)
            best_valid = val_loss
        
        mae, corr, mult_a7, mult_a5, f_score, acc_2 = eval_mosei_senti(results, truths, True) 

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": val_loss,
                    "test_acc_2": acc_2,
                    "test_mult_5": mult_a5,
                    "test_mult_7": mult_a7,
                    "test_mae": mae,
                    "test_corr": corr,
                    "test_f_score": f_score,
                    "best_valid_loss": best_valid,
                }
            )
        )

    model = load_model(hyp_params, name='Unimodal'+"_"+hyp_params.modality)
    _, results, truths = evaluate(model, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')