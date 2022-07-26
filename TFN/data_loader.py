import numpy as np
from tqdm import tqdm_notebook

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import *

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    @property
    def tva_dim(self):
        t_dim = 300
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    

def get_loader(hp, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    
    print(config.mode)
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim
    
    if config.mode == 'train':
        hp.n_train = len(dataset)
    elif config.mode == 'valid':
        hp.n_valid = len(dataset)
    elif config.mode == 'test':
        hp.n_test = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''

        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)
        labels = []
        ids = []

        for sample in batch:
            labels.append(torch.from_numpy(sample[1]))
            ids.append(sample[2])
        labels = torch.cat(labels, dim=0)
        
        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:,0][:,None]

        # Rewrite this
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

        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch],padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        texts = []
        for sample in batch:
            sent = []
            for word in sample[0][3]:
                sent.append(np.array(dataset.pretrained_emb[dataset.word2id[word]]))
            texts.append(torch.as_tensor(sent))
        texts = pad_sequence(texts, target_len=50, batch_first=True, padding_value=PAD)

        return texts, visual, acoustic, labels, ids

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader