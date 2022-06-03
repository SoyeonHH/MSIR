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

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import DiffLoss, MSE, SIMSE, CMD

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
        # sys.stdout = open('/mnt/soyeon/workspace/multimodal/MISA/MISA_mosei.txt', 'w')
        self.model.eval()
        segment_list = []
        words_list = []
        preds = []
        preds_2 = []
        preds_7 = []
        labels = []
        labels_2 = []
        labels_7 = []

        # pkl data
        with open(f"../datasets/MOSEI/mosei.pkl", "rb") as handle:
            data = pickle.load(handle)

        test_data = data["test"]
        test_dataset = get_dataset(test_data)

        test_dataloader = DataLoader(
            test_dataset, batch_size=128, shuffle=False,
        )

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

            words_list.append(words)
            labels.append(label[0][0])

            # label_2 appending
            if label > 0:
                labels_2.append('positive')
            else:
                labels_2.append('negative')
            
            # label_7 appending
            if label < -15/7:
                labels_7.append('very negative')
            elif label < -9/7:
                labels_7.append('negative')
            elif label < -3/7:
                labels_7.append('slightly negative')
            elif label < 3/7:
                labels_7.append('Neutral')
            elif label < 9/7:
                labels_7.append('slightly positive')
            elif label < 15/7:
                labels_7.append('positive')
            else:
                labels_7.append('very positive')

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

        count = 0
        # for i in range(len(segment_list)):
        #     print(i, "th data")
        #     print(segment_list[i])
        #     print(words_list[i])
        #     print(labels[i])
        #     print(labels_2[i])
        #     print(labels_7[i])
        #     print(preds[i])
        #     print(preds_2[i])
        #     print(preds_7[i])

        # sys.stdout.close()
        return segment_list, preds, preds_2, preds_7


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask):
        self.sentences = sentences
        self.visual = visual
        self.acoustic = acoustic
        self.labels = labels
        self.lengths = lengths
        self.bert_sentences = bert_sentences
        self.bert_sentence_types = bert_sentence_types
        self.bert_sentence_att_mask = bert_sentence_att_mask

    

def get_dataset(data):
    bert_details = []
    sentences = []
    visuals = []
    acoustics = []
    labels = []

    for (ex_index, example) in enumerate(data):

        (words, visual, acoustic), label_id, segment = example

        CACHE_PATH = '/mnt/soyeon/workspace/multimodal/MSIR/datasets/MOSEI/embedding_and_mapping.pt'
        pretrained_emb, word2id = torch.load(CACHE_PATH)

        word_ids = []
        aligned_visual = []
        aligned_audio = []
        for idx, word in enumerate(words):
            word_ids.append(word2id[word])
            aligned_visual.append(visual[idx, :])
            aligned_audio.append(acoustic[idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(word_ids) > max_seq_length - 2:
            word_ids = word_ids[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]
            words = words[:max_seq_length - 2]

        # Prepare input
        segment_ids = [0] * len(word_ids)

        pad_length = max_seq_length - len(word_ids)

        acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
        acoustic = np.concatenate((acoustic, acoustic_padding))

        visual_padding = np.zeros((pad_length, VISUAL_DIM))
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * pad_length

        # Pad inputs
        word_ids += padding
        segment_ids += padding

        # Check input length
        assert len(word_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert acoustic.shape[0] == max_seq_length
        assert visual.shape[0] == max_seq_length

        sentences.append(word_ids)
        visuals.append(visual)
        acoustics.append(acoustic)
        labels.append(label_id)

        text = " ".join(words)
        encoded_bert_sent = bert_tokenizer.encode_plus( 
            text, max_length=max_seq_length+2, add_special_tokens=True, pad_to_max_length=True)
        bert_details.append(encoded_bert_sent)

        assert len(bert_details) == max_seq_length

    sentences = pad_sequence([torch.LongTensor(sentence) for sentence in sentences], padding_value=PAD)
    visuals = pad_sequence([torch.FloatTensor(visual) for visual in visuals])
    acoustics = pad_sequence([torch.FloatTensor(acoustic) for acoustic in acoustics])
    labels = torch.FloatTensor([label for label in labels])
    lengths = torch.LongTensor([sentence.shape[0] for sentence in sentences])

    # Bert things are batch_first
    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
    

    dataset = TensorDataset(
        sentences,
        visuals,
        acoustics,
        labels,
        lengths,
        bert_sentences,
        bert_sentence_types,
        bert_sentence_att_mask
    )
    return dataset



class TestMOSI(object):
    def __init__(self, model):
        self.model = model

    def start(self):
        # sys.stdout = open('/mnt/soyeon/workspace/multimodal/MMIM/MMIM_mosi.txt', 'w')
        self.model.eval()
        segment_list = []
        preds = []
        preds_2 = []
        preds_7 = []

        # pkl data
        with open(f"../datasets/MOSI/mosi.pkl", "rb") as handle:
            mosi_data = pickle.load(handle)

        test_data = mosi_data["test"]
        test_data_loader = get_loader(test_data)

        segment_list = [data[2] for data in test_data]

        # prediction
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_data_loader)):

                text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                
                device = torch.device('cuda')
                text, audio, vision, y = text.to(device), audio.to(device), vision.to(device), y.to(device)
                lengths = lengths.to(device)
                bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)

                _, _, outputs, _, _ = self.model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

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

        # for i in range(len(segment_list)):
        #     print(i, "th data")
        #     print(segment_list[i])
        #     print(preds[i])
        #     print(preds_2[i])
        #     print(preds_7[i])

        # sys.stdout.close()

        return segment_list, preds, preds_2, preds_7


def get_loader(dataset):

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        # batch = sorted(batch, key=lambda x: np.array(x[0][0]).shape[0], reverse=True)
        
        # CACHE_PATH = '/home/ubuntu/soyeon/MSIR/datasets/MOSI/embedding_and_mapping.pt'
        CACHE_PATH = '/mnt/soyeon/workspace/multimodal/MSIR/datasets/MOSI/embedding_and_mapping.pt'
        pretrained_emb, word2id = torch.load(CACHE_PATH)

        v_lens = []
        a_lens = []
        labels = []
        ids = []

        sentences = []
        for sample in batch:
            # sentence = np.asarray([word2id[word] for word in sample[0][0]])
            sentence = torch.LongTensor([word2id[word] for word in sample[0][0]])
            sentences.append(sentence)

        for sample in batch:
            if len(sample[0]) > 4: # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][0])]))
                a_lens.append(torch.IntTensor([len(sample[0][0])]))
            labels.append(torch.from_numpy(sample[1]))
            ids.append(sample[2])
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)
        
        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:,0][:,None]


        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor
                    
        # labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        # sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        sentences = pad_sequence(sentences, padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], target_len=alens.max().item())

        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][0])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN+2, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)


        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])


        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn)

    return data_loader