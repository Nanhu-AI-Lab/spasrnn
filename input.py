'''
Dealing with Input stuff, prepare datasets.
'''
import os
import pandas as pd
import torch
import torchtext
# from torchtext.datasets import IMDB, AG_NEWS
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
# from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inputs():
    '''
    Inputs class
    '''
    def __init__(self, cls_dataset):
        self.data_set = cls_dataset()
        self.train_set = cls_dataset(split='train')
        pass

    def dataset(self, split_ratio, min_freq, batch_size, shuffle_test=True):
        # download the dataset
        train_iter, test_iter = self.data_set
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        num_train = int(len(train_dataset) * split_ratio)
        split_train_, split_valid_ = \
            random_split(train_dataset,
                         [num_train,
                         len(train_dataset) - num_train])

        # tokenise the input samples
        self.tokenizer = get_tokenizer('basic_english')
        train_iter = self.train_set

        # build the vocabulary from the training set
        self.vocab = build_vocab_from_iterator(self.yield_tokens(train_iter),
                                               min_freq=min_freq,
                                               specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<pad>'])

        # # build index_word dictionary
        # dic = {}
        # for item in self.vocab.get_stoi():
        #     dic[self.vocab[item]] = item

        train_dataloader = DataLoader(split_train_, batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_batch)
        valid_dataloader = DataLoader(split_valid_, batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=shuffle_test,
                                     collate_fn=self.collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def text_pipeline(self, x):
        return self.vocab(self.tokenizer(x))

    def label_pipeline(self, x):
        if x == 'pos':
            return 1
        else:
            return 0

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (label, text) in batch:
            label_list.append(self.label_pipeline(label))
            processed_text = torch.tensor(self.text_pipeline(text),
                                          dtype=torch.int64)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
        return label_list.to(device), text_list.to(device)

    def word_embeddings(self, glove_name, glove_dim):
        # download the pre-trained word embeddings
        try:
            glove = torchtext.vocab.GloVe(name=glove_name, dim=glove_dim)
        except KeyError:
            glove = torchtext.vocab.GloVe(name='6B', dim=glove_dim)

        matrix_len = len(self.vocab)
        weights_matrix = np.zeros((matrix_len, glove_dim))
        words_found = 0

        for i in range(matrix_len):
            try:
                word = self.vocab.lookup_token(i)
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6,
                                                     size=(glove_dim,))

        weights_matrix = torch.tensor(weights_matrix)

        glove_offset = -1 * torch.min(weights_matrix)
        glove_max = torch.max(weights_matrix) + glove_offset
        # glove_offset, glove_max
        glove_matrix = (weights_matrix + glove_offset) / glove_max
        glove_matrix[0:2, :] = 0

        return glove_matrix


class IMDBInputs(Inputs):
    '''
    IMDB inputs label pipeline
    '''
    # def __init__(self):
    #     # self.data_instance=IMDB()
    #     pass
    def label_pipeline(self, x):
        if x == 'pos':
            return 1
        else:
            return 0


class AGNewsInputs(Inputs):
    '''
    agnews inputs label pipeline
    '''
    # def __init__(self):
    #     # self.data_instance=AG_NEWS()
    #     pass

    def label_pipeline(self, x):
        if x == 1:
            return 0
        if x == 2:
            return 1
        if x == 3:
            return 2
        if x == 4:
            return 3


class SICKDataset(torch.utils.data.IterableDataset):
    '''
    SICK dataset
    '''
    def __init__(self, data):
        super(SICKDataset).__init__()
        self.data_iter = data
        # pd.read_csv(data, iterator=True, header=None, chunksize=1)

    def __iter__(self):
        for data in self.data_iter:
            yield data


# class SICKDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         sick_folder = os.getcwd()
#         sick_fn_en = os.path.join(sick_folder,
#                                   'data',
#                                   'nlp_datasets',
#                                   'SICK.txt')
#         self.data = pd.read_csv(sick_fn_en)
#
#         x = self.data.iloc[:, 0:1].values
#         y = self.data.iloc[:, 2].values
#
#         #self.x_train = torch.tensor(x, dtype=torch.float32)
#         self.y_train = torch.tensor(y, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.y_train)
#
#     def __getitem__(self, idx):
#         return self.x_train[idx], self.y_train[idx]


class SICK():
    '''
    SICK dataset
    '''
    def __init__(self, sick_fn: str):
        self.sick_fn = sick_fn
        self.name = self.sick_fn.split('/')[-1].split('.')[0]
        self.data = self.load_data()
        # self.train_data, self.dev_data, self.test_data = self.split_data()
        self.train_data, self.dev_data, self.test_data = self.split_data()
        self.iterator_train = SICKDataset(self.train_data)
        # iterator_dev = SICKDataset(self.dev_data)
        self.iterator_test = SICKDataset(self.test_data)
        # self.iterator_train, self.iterator_test

    def load_data(self):
        with open(self.sick_fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()][1:]
        lines = [tuple(ln[1:5] + ln[-1:]) for ln in lines]
        lines = [(s1, s2, el, float(rl), split)
                 for (s1, s2, el, rl, split) in lines]
        return lines

    def split_data(self):
        train_data, dev_data, test_data = [], [], []
        for (s1, s2, el, rl, s) in self.data:
            if s == 'TRAIN':
                train_data.append((s1, s2, el, rl))
            if s == 'TRIAL':
                dev_data.append((s1, s2, el, rl))
            if s == 'TEST':
                test_data.append((s1, s2, el, rl))
        return train_data, dev_data, test_data


class SICKInputs(Inputs):
    '''
    SICK dataset input
    '''
    def __init__(self):
        super(SICKInputs).__init__()
        self.sick_folder = os.getcwd()
        self.sick_fn_en = os.path.join(self.sick_folder,
                                       'data',
                                       'nlp_datasets',
                                       'SICK.txt')
        self.data = SICK(self.sick_fn_en)

    def yield_tokens(self,data_iter):
        for s1, s2, _, _ in data_iter:
            yield self.tokenizer(' '.join((s1,s2)))
            # self.tokenizer(str.join(s1, s2))

    def text_pipeline(self,x):
        return self.vocab(self.tokenizer(x))

    def label_pipeline(self, x):
        if x == 'ENTAILMENT':
            return 1
        if x == 'NEUTRAL':
            return 2
        else:
            return 0

    def collate_batch(self, batch):
        label_list, s1_list, s2_list = [], [], []
        for (s1, s2, label, _) in batch:
            label_list.append(self.label_pipeline(label))
            processed_s1 = torch.tensor(self.text_pipeline(s1),
                                        dtype=torch.int64)
            processed_s2 = torch.tensor(self.text_pipeline(s2),
                                        dtype=torch.int64)
            s1_list.append(processed_s1)
            s2_list.append(processed_s2)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        s1_list = pad_sequence(s1_list, batch_first=True, padding_value=0)
        s2_list = pad_sequence(s2_list, batch_first=True, padding_value=0)
        return label_list.to(device), s1_list.to(device), s2_list.to(device)

    def dataset(self,split_ratio, min_freq, batch_size, shuffle_test=True):
        train_iter, test_iter = self.data.train_data,self.data.test_data
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        num_train = int(len(train_dataset) * split_ratio)
        split_train_, split_valid_ = random_split(train_dataset,
                                                  [num_train,
                                                  len(train_dataset) - \
                                                    num_train])

        # tokenise the input samples
        self.tokenizer = get_tokenizer('basic_english')
        train_iter = self.data.train_data

        # build the vocabulary from the training set
        self.vocab = build_vocab_from_iterator(self.yield_tokens(train_iter),
                                               min_freq=min_freq,
                                               specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<pad>'])

        train_dataloader = DataLoader(split_train_, batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_batch)
        valid_dataloader = DataLoader(split_valid_, batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=shuffle_test,
                                     collate_fn=self.collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader


class YelpDataset(torch.utils.data.Dataset):
    '''
    Yelp dataset
    '''
    def __init__(self,path):
        super(YelpDataset).__init__()
        self.data = pd.read_csv(path)

        self.label = self.data.iloc[:, [0]].values.tolist()
        self.sentence = self.data.iloc[:, [1]].values.tolist()

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return (self.label[idx][0], self.sentence[idx][0])

class YelpInputs(Inputs):
    '''
    Yelp dataset input
    '''
    def __init__(self):
        super(YelpInputs).__init__()
        self.yelp_folder = os.getcwd()
        self.yelp_train_file_path = os.path.join(self.yelp_folder,
                                                 'data',
                                                 'nlp_datasets',
                                                 'yelp',
                                                 'train.csv')
        self.yelp_test_file_path = os.path.join(self.yelp_folder,
                                                'data',
                                                'nlp_datasets',
                                                'yelp',
                                                'test.csv')
        self.train_data = YelpDataset(self.yelp_train_file_path)
        self.test_data = YelpDataset(self.yelp_test_file_path)

    def dataset(self, split_ratio, min_freq, batch_size, shuffle_test=True):
        # download the dataset
        train_iter, test_iter = self.train_data, self.test_data

        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        #train_dataset, test_dataset = self.train_data, self.test_data
        num_train = int(len(train_dataset) * split_ratio)
        split_train_, split_valid_ = random_split(train_dataset,
                                                  [num_train,
                                                  len(train_dataset) - \
                                                    num_train])
        #Dist_plot_of_sequence_length(split_train_)

        # tokenise the input samples
        self.tokenizer = get_tokenizer('basic_english')
        train_iter = self.train_data

        # build the vocabulary from the training set
        self.vocab = build_vocab_from_iterator(self.yield_tokens(train_iter),
                                               min_freq=min_freq,
                                               specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<pad>'])

        train_dataloader = DataLoader(split_train_, batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_batch)
        valid_dataloader = DataLoader(split_valid_, batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=shuffle_test,
                                     collate_fn=self.collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def text_pipeline(self, x):
        return self.vocab(self.tokenizer(x))

    def label_pipeline(self, x):
        if x == 1:
            return 0
        if x == 2:
            return 1
        if x == 3:
            return 2
        if x == 4:
            return 3
        if x == 5:
            return 4

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (label, text) in batch:
            label_list.append(self.label_pipeline(label))
            processed_text = torch.tensor(self.text_pipeline(text),
                                          dtype=torch.int64)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
        return label_list.to(device), text_list.to(device)

    def word_embeddings(self, glove_name, glove_dim):
        # download the pre-trained word embeddings
        try:
            glove = torchtext.vocab.GloVe(name=glove_name, dim=glove_dim)
        except KeyError:
            glove = torchtext.vocab.GloVe(name='6B', dim=glove_dim)

        matrix_len = len(self.vocab)
        weights_matrix = np.zeros((matrix_len, glove_dim))
        words_found = 0

        for i in range(matrix_len):
            try:
                word = self.vocab.lookup_token(i)
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6,
                                                     size=(glove_dim,))

        weights_matrix = torch.tensor(weights_matrix)

        glove_offset = -1 * torch.min(weights_matrix)
        glove_max = torch.max(weights_matrix) + glove_offset
        # glove_offset, glove_max
        glove_matrix = (weights_matrix + glove_offset) / glove_max
        glove_matrix[0:2, :] = 0

        return glove_matrix
