'''
Validation SRNN and plot Attention figure.

Author: Wang Minghao
Date: 2023.1.5
'''
# https://discuss.pytorch.org/t/large-performance-gap-between-pytorch-and-keras-for-imdb-sentiment-analysis-model/135659

# tf dataset+ glove  'can not train successfully'
# import sys
# from matplotlib.pyplot import text
import numpy as np
import torch
# from metrics_helper import Score
from torch import nn
# from tqdm import tqdm

import configparser
from data_loader_util import load_data_imdb_glove, ag_news_load_data_by_glove, yelp_load_data_by_glove
from lib.SRNN.srnn_cuda import SRNNDense2sparse, SNNDense2sparse, SNNReadoutModuleSoftmaxForRecord #,SNNReadoutModule
from torchtext.data.utils import get_tokenizer
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

# remember to change this before the first run
home_dir =  os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(home_dir, 'model/')
fig_dir = os.path.join(home_dir, 'fig/')
con_dir = os.path.join(home_dir, 'config.ini')

con = configparser.ConfigParser()
con.read(con_dir, encoding='utf-8')
opt = dict(dict(con.items('paths')), **dict(con.items("para")))
# train_dataloader,test_dataloader=\
# load_data_imdb_tf(batch_size=batch_size,device=device,maxlen=maxlen)

# use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.manual_seed(int(opt['seed']))
np.random.seed(int(opt['seed']))

out_size = 0

class SPASRNN(nn.Module):
    '''
    sparse sRnn network
    '''
    def __init__(self, wordemb_matrix, hid_dim, out_dim, dropout_prob,
                 srnn_args):
        super().__init__()

        self.vocab_size, self.embedding_dim =\
             wordemb_matrix.shape[0], wordemb_matrix.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': wordemb_matrix})
        self.embedding.weight.requires_grad = True
        self.srnn1 = SRNNDense2sparse(
            self.embedding_dim, hid_dim, hid_dim, srnn_args)
        self.readout = SNNReadoutModuleSoftmaxForRecord(hid_dim, out_dim, srnn_args)

        # fully connected layer for output
        self.fc = torch.nn.Linear(hid_dim*2, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

        print('sRnn Using {} recurrent mode'.format(opt['recurrent_mode']))
        print('Readout Using {} readout mode'.format(opt['readout_mode']))

    def forward(self, feed_data):
        embed_out = self.dropout(self.embedding(feed_data))
        rnn_hid, _ = self.srnn1(embed_out)
        snn_out = self.readout(rnn_hid)

        return snn_out

    def get_dweight_r(self):
        return self.srnn1.getDweightR()


class SPASNN(nn.Module):
    '''
    sparse snn network
    '''
    def __init__(self, wordemb_matrix, hid_dim, out_dim,
                 dropout_prob, srnn_args):
        super().__init__()

        self.vocab_size, self.embedding_dim = \
            wordemb_matrix.shape[0], wordemb_matrix.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': wordemb_matrix})
        self.embedding.weight.requires_grad = True
        self.snn1 = SNNDense2sparse(
            self.embedding_dim, hid_dim, hid_dim, srnn_args)
        self.readout = SNNReadoutModuleSoftmaxForRecord(hid_dim, out_dim, srnn_args)

        # fully connected layer for output
        self.fc = torch.nn.Linear(hid_dim*2, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, feed_data):
        embed_out = self.dropout(self.embedding(feed_data))
        rnn_hid, _ = self.snn1(embed_out)
        snn_out = self.readout(rnn_hid)

        return snn_out


def main():
    global out_size
    # load test dataset
    test_dataloader, _, glove_matrix = dataset_builder(opt['dataset'])

    out_size = get_out_size(opt['dataset'])

    # set resume_id for model save
    if opt['model_name'] == 'd2s_sRnn':
        resume_id = opt['dataset'] + '_' + opt['model_name'] + '_' + \
        opt['recurrent_mode'] + '_' + opt['reset_m'] + '_' + opt['readout_mode']
    elif opt['model_name'] == 'd2s_snn':
        resume_id = opt['dataset'] + '_' + opt['model_name'] + '_' + \
            opt['reset_m'] + '_' + opt['readout_mode']
    elif opt['model_name'] == 'slayer':
        resume_id = opt['dataset'] + '_' + opt['model_name'] + '_' + opt['readout_mode']
    else:
        raise Exception(
            'The model you chose has not been implemented, \
                please select from config.py')

    # switch to LSTM baseline
    # model = LSTM(vocab_size, embedding_dim, hidden_dim)
    print('LIF reset mode is {}'.format(opt['reset_m']))
    if opt['model_name'] == 'd2s_sRnn':
        model = SPASRNN(glove_matrix, hid_dim=int(opt['nb_hidden']),
                        out_dim=out_size,
                        dropout_prob=0.5, srnn_args=opt)
    if opt['model_name'] == 'd2s_snn':
        model = SPASNN(glove_matrix, hid_dim=int(opt['nb_hidden']),
                       out_dim=out_size,
                       dropout_prob=0.5, srnn_args=opt)

    # optimizer = optim.Adam(model.parameters(), \
    # betas=(0.7, 0.995), lr=learning_rate)

    # load model
    model_saved_flag = os.path.isfile(model_dir + resume_id + '.pt')
    if not model_saved_flag:
        raise Exception(
            'The requered model file not exist, \
                please check or train from scrach')
    else:
        print('==> Load model from file..')
        saved_model = torch.load(
            model_dir + resume_id + '.pt', map_location=opt['device'])
        model.load_state_dict(saved_model['model'])
        # optimizer.load_state_dict(saved_model['optimizer'])

    model.to(device=opt['device'])
    # nn.BCELoss()# torch.nn.CrossEntropyLoss()
    # loss_function = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        # Get the aim sentence
        gtzer = get_tokenizer('basic_english')
        aim_sentence_text = gtzer(
            test_dataloader.dataset._data[int(opt['asi'])][1])
        aim_sentence_len = len(aim_sentence_text)

        break_flag = False
        batch_index = 0

        # tm=tqdm(test_dataloader, desc='evaluating...', file=sys.stdout)
        for labels, sentences in test_dataloader:
            aim_sentence_local_index = int(opt['asi']) - \
                                       batch_index * int(opt['b'])

            if 0 <= aim_sentence_local_index < int(opt['b']):
                sentences = sentences.to(opt['device'])
                labels = labels.to(opt['device'])
                score = model(sentences).squeeze(1)
                pred = torch.argmax(torch.softmax(score, dim=1), dim=1)

                softmax_list = model.readout.get_softmax_value()
                softmax_tensor = torch.zeros(
                    (aim_sentence_len, out_size))
                for i in range(len(softmax_list)):
                    if i < aim_sentence_len:
                        softmax_tensor[i, :] = \
                            softmax_list[i +1][aim_sentence_local_index]
                    else:
                        break

                aim_sentence_pred = pred[aim_sentence_local_index]
                aim_sentence_label = labels[aim_sentence_local_index]
                print(
                    f'Aimed snetence True label is {aim_sentence_label:d}, \
                        prediction is {aim_sentence_pred:d}')

                plot_attention_visualization(dataset_name=opt['dataset'],
                                             original_text=aim_sentence_text,
                                             weight_data=softmax_tensor,
                                             true_label=aim_sentence_label,
                                             pred_label=aim_sentence_pred,
                                             save_dir=fig_dir)

                break_flag = True
            else:
                model.readout.clear_softmax_value()
                batch_index += 1
            # softmax_value_after = model.readout.get_softmax_value()

            # cal_score = Score(y_true=labels.detach().cpu().numpy(),
            #               y_pred=pred.detach().cpu().numpy(),
            #               average='macro')
            # acc = cal_score.cal_acc()
            # f1 = cal_score.cal_f1()
            # precision = score.cal_precision()
            # recall = score.cal_recall()
            if break_flag:
                break


def get_out_size(dataset_name):
    if dataset_name == 'agnews':
        out_size_num = 4
    elif dataset_name == 'imdb':
        out_size_num = 2
    elif dataset_name == 'yelp':
        out_size_num = 5
    else:
        raise Exception('Please check the dataset you use, and set the out size in this function.')
    
    return out_size_num

def dataset_builder(dataset):
    if dataset == 'agnews':
        train_dataloader, test_dataloader, _, glove_matrix = \
            ag_news_load_data_by_glove(batch_size=int(opt['b']))
    elif dataset == 'imdb':
        train_dataloader, test_dataloader, _, glove_matrix = \
            load_data_imdb_glove(batch_size=int(opt['b']))
    elif dataset == 'yelp':
        train_dataloader, test_dataloader, _, glove_matrix = \
            yelp_load_data_by_glove(batch_size=int(opt['b']))
    else:
        raise Exception('The {0} dataset is not supported for now.'.format(dataset))
    return test_dataloader, train_dataloader, glove_matrix


def plot_attention_visualization(dataset_name: str,
                                 original_text: list,
                                 weight_data: torch.tensor,
                                 true_label: int,
                                 pred_label: int,
                                 save_dir: str):
    '''
    Plot the attention of a sentence with the readout weight

    Author: Zhang Qian
    Modified: Wang Minghao
    Date: 2022.11.09
    '''

    # make dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Set label according to dataset
    if dataset_name == 'agnews':
        labels = ['0.World', '1.Sports', '2.Business', '3.Sci/Tech']
    elif dataset_name == 'imdb':
        labels = ['0.negative', '1.positive']
    else:
        raise Exception('{} is not implemented, for now only support imdb and agnews datasets.'.format(dataset_name))

    # 需要显示的语句内容
    word_single = list(original_text)
    # 需要显示的标签
    label_single = list(labels)
    h_or_v = 'h'
    if h_or_v == 'h':
        d = np.array(weight_data.T)
        df = pd.DataFrame(d, columns=word_single, index=label_single)  # 横文本
        plt.rcParams['figure.figsize'] = (20, 1)  # 横文本
    elif h_or_v == 'v':
        d = np.array(weight_data)
        df = pd.DataFrame(d, columns=label_single, index=word_single)  # 竖文本
        plt.rcParams['figure.figsize'] = (4, 12)  # 竖文本

    # set plt params
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    fontsize = 12

    # 生成空白图形赋值给fig对象
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='Wistia', aspect='auto')
    #cax = ax.matshow(df)
    fig.colorbar(cax, shrink=1)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    rotation = 45
    # 设置文字旋转
    # fontdict = {'rotation': 'vertical'}
    # 或者这样设置文字旋转
    # fontdict = {'rotation': 45}
    # 或者直接设置到这里
    # ax.set_xticklabels([''] + list(df.columns), rotation=rotation)
    # 或者直接设置到这里
    # fig.autofmt_xdate(rotation=rotation)

    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()],
            rotation=rotation,
             ha='left', va='center', rotation_mode='anchor')

    # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
    ax.set_xticklabels([''] + list(df.columns), fontsize=fontsize)
    ax.set_yticklabels([''] + list(df.index), fontsize=fontsize)

    fig.savefig(save_dir + '/' + dataset_name +\
                '_attention_visualization_{0}_\
                    TrueLabel{1}_PredLable{2}_r{3}.png'.\
                format(int(opt['asi']),
                       true_label,
                       pred_label,
                       rotation),
                bbox_inches='tight', pad_inches=0.3)


if __name__ == '__main__':
    main()
