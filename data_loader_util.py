"""
load data
"""
import numpy as np
from keras.datasets import imdb
from tensorflow.keras import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchtext.datasets import IMDB, AG_NEWS
import configparser
import os

from input import IMDBInputs, AGNewsInputs, SICKInputs, YelpInputs

home_dir = os.path.dirname(os.path.realpath(__file__))
con_dir = os.path.join(home_dir, 'config.ini')

con = configparser.ConfigParser()
con.read(con_dir,encoding='utf-8')
opt = dict(dict(con.items('paths')),**dict(con.items("para")))

glove_name = opt['glove_name'] # 6B
glove_dim = int(opt['nb_embedding']) # 100
split_ratio = float(opt['split_ratio']) # 0.7
min_freq = int(opt['min_freq']) # 10
def load_data_imdb_tf(batch_size,device,maxlen):
    vocab_size = 10000
    print('tensor flow imdb loading data..')
    (train_data, train_labels), (test_data, test_labels) = \
                        imdb.load_data(num_words=vocab_size)
    x_train = preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(test_data, maxlen=maxlen)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    train_data = list(zip(x_train, y_train))
    test_data = list(zip(x_test, y_test))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, \
                        shuffle=True, collate_fn=lambda x: tuple(x_.to(device) \
                        for x_ in default_collate(x)))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, \
                        shuffle=True,collate_fn=lambda x: tuple(x_.to(device) \
                        for x_ in default_collate(x)))

    return train_dataloader,test_dataloader


def load_data_imdb_glove(batch_size, shuffle_test=True):
    print('loading imdb data...')
    ds_cls=IMDB
    input_data = IMDBInputs(ds_cls)

    train_dataloader, valid_dataloader, test_dataloader = \
            input_data.dataset(split_ratio, min_freq,
                               batch_size, shuffle_test=shuffle_test)
    glove_matrix = input_data.word_embeddings(glove_name, glove_dim)
    # train_ldr = torch.utils.data.DataLoader(dataset=train_data,
    #                           batch_size=batch_size, shuffle=True, \
    #                           drop_last=True,num_workers=0,\
    #                           collate_fn=lambda x: tuple(x_.to(device) \
    #                           for x_ in default_collate(x)))
    # test_ldr = torch.utils.data.DataLoader(dataset=test_data,
    #                           batch_size=batch_size, shuffle=True, \
    #                           drop_last=True, num_workers=0,\
    #                           collate_fn=lambda x: tuple(x_.to(device) \
    #                           for x_ in default_collate(x)))
    return train_dataloader,test_dataloader,valid_dataloader,glove_matrix

def ag_news_load_data_by_glove(batch_size, shuffle_test=True):
    print('loading ag news data...')
    ds_cls = AG_NEWS
    input_data = AGNewsInputs(ds_cls)
    train_dataloader, valid_dataloader, test_dataloader = \
            input_data.dataset(split_ratio, min_freq,
                               batch_size, shuffle_test=shuffle_test)
    glove_matrix = input_data.word_embeddings(glove_name, glove_dim)
    return train_dataloader,test_dataloader,valid_dataloader,glove_matrix

def yelp_load_data_by_glove(batch_size, shuffle_test=True):
    print('loading yelp data...')
    input_data = YelpInputs()
    train_dataloader, valid_dataloader, test_dataloader = \
            input_data.dataset(split_ratio, min_freq,
                               batch_size, shuffle_test=shuffle_test)
    glove_matrix = input_data.word_embeddings(glove_name, glove_dim)
    return train_dataloader,test_dataloader,valid_dataloader,glove_matrix

def sick_load_data_by_glove(batch_size, shuffle_test=True):
    print('loading sick data...')
    input_data = SICKInputs()
    train_dataloader, valid_dataloader, test_dataloader =\
            input_data.dataset(split_ratio, min_freq,
                               batch_size, shuffle_test=shuffle_test)
    glove_matrix = input_data.word_embeddings(glove_name, glove_dim)
    return train_dataloader,test_dataloader,valid_dataloader,glove_matrix


#unit test
#data=load_data_ag_news(64)
#data=load_data_imdb_glove(64)
#load_data_imdb_tf()
# train,test,valid=sick_load_data_by_glove(64)
# for i, (labels, s1, s2) in enumerate(test):
#     print(s1, labels)
#     # print(s2,labels)
#     break
