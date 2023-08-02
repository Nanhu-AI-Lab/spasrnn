'''
make a comparison with slayer snn, sparse srnn, sparse snn
'''
import os
import torch
from torch import nn
import slayerSNN as slsnn
import configparser
from lib.SRNN.srnn_cuda import SNNReadoutModule, SRNNDense2sparse, SNNDense2sparse

CURRENT_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
netParams = slsnn.params(CURRENT_TEST_DIR + '/slayer_network.yaml')

home_dir = os.path.dirname(os.path.realpath(__file__))
con_dir = os.path.join(home_dir, 'config.ini')

con = configparser.ConfigParser()
con.read(con_dir,encoding='utf-8')

opt = dict(dict(con.items('paths')),**dict(con.items("para")))

class SlayerSNN(nn.Module):
    '''
    switch to slayer snn
    '''
    def __init__(self, wordemb_matrix, hid_dim, out_dim, dropout_prob):
        super().__init__()
        slayer = slsnn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        self.vocab_size, self.embedding_dim = \
            wordemb_matrix.shape[0], wordemb_matrix.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': wordemb_matrix})
        self.embedding.weight.requires_grad = True
        self.snn_fc1=slayer.dense(self.embedding_dim,hid_dim)
        self.snn_fc2=slayer.dense(hid_dim, out_dim)
        # if srnnargs.readout_mode == 'mean':
        #     self.readout = SNNReadout_Module(hid_dim, out_dim, srnnargs)
        # elif srnnargs.readout_mode == 'softmax':
        #     self.readout = SNNReadout_Module_softmax(hid_dim, out_dim, \
        #                                               srnnargs)
        # else:
        #     raise Exception('Please check the readout mode in config.py, \
        #                       using mean or softmax')
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, feed_data):
        embed_out = self.dropout(self.embedding(feed_data))
        self.slayer.simulation['tSample'] = embed_out.size(1)
        # netParams['training']['error']['tgtSpikeRegion']['stop'] = \
        #                                             embed_out.size(1)
        # change tensor form [B,T,N] -> [B,C,H,W,T] == ([B,N,1,1,T])
        embed_out = \
                embed_out.transpose(1,2).unsqueeze(dim=2).unsqueeze(dim=2)
        hid_out = self.slayer.spike(self.slayer.psp(self.snn_fc1(embed_out)))
        snn_out = self.slayer.spike(self.slayer.psp(self.snn_fc2(hid_out)))
        return snn_out
        # # change tensor form [B,C,H,W,T] == ([B,N,1,1,T]) -> [B,T,N]
        # hid_out = hid_out.squeeze(dim=2).squeeze(dim=2).transpose(1,2)
        # snn_out = self.readout(hid_out)

        # if srnnargs.readout_mode == 'mean':
        #     return (snn_out.mean(1))
        # elif srnnargs.readout_mode == 'softmax':
        #     return snn_out
        # else:
        #     raise Exception('Please check the readout mode in config.py, \
        #                       using mean or softmax')

class SPASRNN(nn.Module):
    '''
    switch to sparse srnn
    '''
    def __init__(self, wordemb_matrix, hid_dim, out_dim, dropout_prob):
        super().__init__()

        self.vocab_size, self.embedding_dim = wordemb_matrix.shape[0], \
                                                wordemb_matrix.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': wordemb_matrix})
        self.embedding.weight.requires_grad = True
        self.srnn1=SRNNDense2sparse(self.embedding_dim,hid_dim, hid_dim, \
                                    opt)
        self.readout = SNNReadoutModule(hid_dim, out_dim, opt)

        # fully connected layer for output
        self.fc = torch.nn.Linear(hid_dim*2, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

        print('sRnn Using {0} recurrent mode'.format(opt['recurrent_mode']))
        print('Readout Using {0} readout mode'.format(opt['readout_mode']))

    def forward(self, feed_data):
        embed_out = self.dropout(self.embedding(feed_data))
        rnn_hid, _ = self.srnn1(embed_out)
        snn_out = self.readout(rnn_hid)

        return snn_out

    def get_dweight_r(self):
        return self.srnn1.get_dweight_r()

class SPASNN(nn.Module):
    '''
    switch to sparse snn
    '''
    def __init__(self, wordemb_matrix, hid_dim, out_dim, dropout_prob):
        super().__init__()

        self.vocab_size, self.embedding_dim = wordemb_matrix.shape[0], \
                                                wordemb_matrix.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': wordemb_matrix})
        self.embedding.weight.requires_grad = True
        self.snn1=SNNDense2sparse(self.embedding_dim,hid_dim, hid_dim, \
            opt)
        self.readout = SNNReadoutModule(hid_dim, out_dim, opt)

        # fully connected layer for output
        self.fc = torch.nn.Linear(hid_dim*2, out_dim)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, feed_data):
        embed_out = self.dropout(self.embedding(feed_data))
        rnn_hid, _ = self.snn1(embed_out)
        snn_out = self.readout(rnn_hid)

        return snn_out
