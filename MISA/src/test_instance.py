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

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, to_cpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models

DEVICE = torch.device("cuda:0")

# MOSI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 47
TEXT_DIM = 768

# MOSEI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 35
# TEXT_DIM = 768

max_seq_length = 50
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class TestMOSEI(object):
    def __init__(self, model):
        self.model = model

    def start(self):
        # sys.stdout = open('/home/ubuntu/soyeon/MSIR/MISA/MISA_mosei.txt', 'w')
        self.model.eval()
        segment_list = []
        preds = []
        preds_2 = []
        preds_7 = []

        # pkl data
        with open(f"../../datasets/MOSEI/mosei.pkl", "rb") as handle:
            data = pickle.load(handle)

        test_data = data["test"]
        test_dataloader = get_loader(test_data)

        video = set()
        count = 0
        
        for idx in range(len(test_data)):
            (words, visual, acoustic), label, segment = test_data[idx]
            # if args.dataset == 'mosi':
            # segment_list.append(segment)
            # else:
            video_name = segment[0]
            if video_name in video:
                count += 1
            else:
                video.add(video_name)
                count = 0
            segment_list.append(video_name + '[' + str(count) + ']')

        # prediction
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader)):
                self.model.zero_grad()

                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                # l = to_gpu(l)
                l = to_cpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                outputs = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                # logits = outputs[0]
                logits = outputs.detach().cpu().numpy()
                logits = np.squeeze(logits).tolist()

                preds.extend(logits)

                for logit in logits:
                    # preds_2 appending
                    if logit > 0:
                        preds_2.append('positive')
                    else:
                        preds_2.append('negative')

                    # label_7 appending
                    if logit < -15/7:
                        preds_7.append('very negative')
                    elif logit < -9/7:
                        preds_7.append('negative')
                    elif logit < -3/7:
                        preds_7.append('slightly negative')
                    elif logit < 3/7:
                        preds_7.append('Neutral')
                    elif logit < 9/7:
                        preds_7.append('slightly positive')
                    elif logit < 15/7:
                        preds_7.append('positive')
                    else:
                        preds_7.append('very positive')

        assert len(segment_list) == len(preds)

        for i in range(len(segment_list)):
            print(i, "th data")
            print(segment_list[i])
            print(preds[i])
            print(preds_2[i])
            print(preds_7[i])

        # sys.stdout.close()
        return segment_list, preds, preds_2, preds_7




class TestMOSI(object):
    def __init__(self, model):
        self.model = model

    def start(self):
        # sys.stdout = open('/home/ubuntu/soyeon/MSIR/MISA/MISA_mosi.txt', 'w')
        self.model.eval()
        segment_list = []
        preds = []
        preds_2 = []
        preds_7 = []

        # pkl data
        with open(f"../../datasets/MOSI/mosi.pkl", "rb") as handle:
            mosi_data = pickle.load(handle)

        test_data = mosi_data["test"]
        test_data_loader = get_loader(test_data)

        segment_list = [data[2] for data in test_data]

        # prediction
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_data_loader)):
                batch = tuple(t.to(DEVICE) for t in batch)

                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                # l = to_gpu(l)
                l = to_cpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                outputs = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                logits = outputs.detach().cpu().numpy()
                logits = np.squeeze(logits).tolist()

                preds.extend(logits)

                for logit in logits:
                    # preds_2 appending
                    if logit > 0:
                        preds_2.append('positive')
                    else:
                        preds_2.append('negative')

                    # label_7 appending
                    if logit < -15/7:
                        preds_7.append('very negative')
                    elif logit < -9/7:
                        preds_7.append('negative')
                    elif logit < -3/7:
                        preds_7.append('slightly negative')
                    elif logit < 3/7:
                        preds_7.append('Neutral')
                    elif logit < 9/7:
                        preds_7.append('slightly positive')
                    elif logit < 15/7:
                        preds_7.append('positive')
                    else:
                        preds_7.append('very positive')

        assert len(segment_list) == len(preds)

        for i in range(len(segment_list)):
            print(i, "th data")
            print(segment_list[i])
            print(preds[i])
            print(preds_2[i])
            print(preds_7[i])

        # sys.stdout.close()
        return segment_list, preds, preds_2, preds_7


def get_loader(dataset):

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        # batch = sorted(batch, key=lambda x: np.array(x[0][0]).shape[0], reverse=True)
        
        # CACHE_PATH = '/mnt/soyeon/workspace/multimodal/MISA/datasets/MOSEI/embedding_and_mapping.pt'
        CACHE_PATH = '/home/ubuntu/soyeon/MSIR/datasets/MOSI/embedding_and_mapping.pt'
        pretrained_emb, word2id = torch.load(CACHE_PATH)
        
        sentences = []
        for sample in batch:
            # sentence = np.asarray([word2id[word] for word in sample[0][0]])
            sentence = torch.LongTensor([word2id[word] for word in sample[0][0]])
            sentences.append(sentence)
                    
        # labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        # sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        sentences = pad_sequence(sentences, padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        bert_details = []
        labels = []
        for sample in batch:
            text = " ".join(sample[0][0])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)
            bert_details.append(encoded_bert_sent)
            # if sample[1] 모든 요소가 0이라면, labels.append(0) else sample[1]에서 null 값은 모두 없앤 후 labels.append(sample[1][0])
            if sample[1].all() == 0.:
                labels.append([sample[1][0][0]])
            else:
                labels.append([np.nan_to_num(sample[1])[0][0]])

        # if labels.size(1) == 7:
        #     labels = labels[:,0][:,None]

        labels = torch.cat([torch.FloatTensor(label) for label in labels], dim=0)

        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])


        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])

        return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn)

    return data_loader