import torch
from torch import nn
import sys
from src import models_test
from src.utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from src.eval_metrics import *

# Construct the model

def initiate(hyp_params, test_loader):
    model = getattr(models_test, hyp_params.model+'Model')(hyp_params)
    device = hyp_params.device

    if hyp_params.use_cuda:
        model = model.to(device)
    
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    optimizer.param_groups[0]['capturable'] = True
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return tester(settings, hyp_params, test_loader)


def tester(settings, hyp_params, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    device = hyp_params.device

    def save_tester_attention_score(model, criterion):
        model.eval()
        
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(test_loader):
                sample_index, text, audio, vision = batch_X
                tester_attr = batch_Y.squeeze(dim=-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, tester_attr = \
                            text.to(device), audio.to(device), vision.to(device), tester_attr(device)
                        
                batch_size = text.size(0)

                net = nn.DataParallel(model) if batch_size > 10 else model
                pred, proj_x_a, proj_x_l, proj_x_v, 