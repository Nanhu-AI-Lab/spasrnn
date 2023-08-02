'''
SRNN cuda file

Author: Wang Kun
'''
# from lib2to3.pgen2.token import LPAR
# from logging import basicConfig
# from turtle import forward
# from unicodedata import bidirectional
import numpy as np
import torch
from torch import nn
# import math
import torch.nn.functional as F
# from .spike_neuron import *
from lib.SRNN.snn_layers import SNNReadout as snn_readout
from lib.SRNN.dense2sparse import Dense2SparseSRNNLayer, Dense2SparseSNNLayer
import s3gd_cuda
import time


surr_grad_spike_cuda = s3gd_cuda.surr_grad_spike
s3gd_backward_cuda = s3gd_cuda.s3gd_w_backward
s3gd_s_backward_master = s3gd_cuda.s3gd_s_backward_master
s3gd_wR_backward_cuda = s3gd_cuda.s3gd_wR_backward
# s3gd_wb_backward_cuda = s3gd_cuda.s3gd_wb_backward_cuda
# s3gd_wRb_backward_cuda = s3gd_cuda.s3gd_wRb_backward_cuda


class SNNReadoutModule(nn.Module):
    '''
    SNN Readout Module layer, including mean and softmax mode, set in args.
    '''
    def __init__(self, nb_hidden, nb_outputs, args):
        super(SNNReadoutModule, self).__init__()

        self.params = args
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.snn_readout = snn_readout
        self.weight_end = torch.empty((self.nb_hidden, self.nb_outputs),
                                       device='cuda', dtype=torch.float,
                                       requires_grad=True)
        # torch.nn.init.normal_(self.weight_end, mean=0.0,
        #                       std=20. * (1.0 - float(np.exp(-0.1))) /\
        #                          np.sqrt(self.nb_hidden))
        nn.init.xavier_uniform_(self.weight_end)

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        device = 'cuda'
        dtype = torch.float

        h2 = torch.einsum('abc,cd->abd', (x, self.weight_end))
        mem = torch.zeros((batch_size, self.nb_outputs),
                          device=device, dtype=dtype)
        out = torch.zeros((batch_size, self.nb_outputs),
                          device=device, dtype=dtype)

        alpha = float(np.exp(-0.1))

        if self.params['readout_mode'] == 'mean':
            out_rec = [out]

        for t in range(nb_steps - 1):
            # new_out = prs['beta_readout'] * out + \
            # (1-prs['beta_readout'])*h2[:, t, :]
            mem = alpha * mem + (1 - alpha) * h2[:, t, :]

            if self.params['readout_mode'] == 'mean':
                out = mem
                out_rec.append(out)
            elif self.params['readout_mode'] == 'softmax':
                out = out + F.softmax(mem, dim=1)
            else:
                raise Exception('Please check the readout mode in config.py, \
                    for now, must in mean or softmax')

        if self.params['readout_mode'] == 'mean':
            out_rec = torch.stack(out_rec, dim=1)
            return out_rec.mean(1)
        elif self.params['readout_mode'] == 'softmax':
            return out
        else:
            raise Exception('Please check the readout mode in config.py, \
                for now, must in mean or softmax')


class SNNReadoutModuleSoftmaxForRecord(nn.Module):
    '''
    SNN Readout module which saving the softmax value \
        for attention visualization.
    Note:
        When using this Module, after each batch,
        please remember to call 'clear_softmax_value' method
        to clear the 'softmax_value_rec' list,
        otherwise, will raise Out of Memery (OOM) error.

    Example:
        for b in range(batch):
            output = model(input)
            loss = LossFunction()
            acc = AccFuntion()
            ...

            model.readout..clear_softmax_value()
    '''

    def __init__(self, nb_hidden, nb_outputs, args):
        super(SNNReadoutModuleSoftmaxForRecord, self).__init__()

        self.params = args
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        # self.snn_readout = snn_readout
        self.weight_end = torch.empty((self.nb_hidden, self.nb_outputs),
                                       device='cuda', dtype=torch.float,
                                       requires_grad=True)
        self.softmax_value_rec = []
        # torch.nn.init.normal_(self.weight_end, mean=0.0,
        #                       std=20. * (1.0 - float(np.exp(-0.1))) / \
        #                       np.sqrt(self.nb_hidden))
        nn.init.xavier_uniform_(self.weight_end)
        # print('testing new SNN Readout model')

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        device = 'cuda'
        dtype = torch.float

        h2 = torch.einsum('abc,cd->abd', (x, self.weight_end)
                          ).to(device=device)
        mem = torch.zeros((batch_size, self.nb_outputs),
                          device=device, dtype=dtype)
        out = torch.zeros((batch_size, self.nb_outputs),
                          device=device, dtype=dtype)

        alpha = float(np.exp(-0.1))

        for t in range(nb_steps - 1):
            mem = alpha * mem + (1 - alpha) * h2[:, t, :]
            softmax_value = F.softmax(mem, dim=1)
            self.softmax_value_rec.append(softmax_value)
            out = out + softmax_value

        return out

    def get_softmax_value(self):
        return self.softmax_value_rec

    def clear_softmax_value(self):
        self.softmax_value_rec.clear()


class SRNNDense2sparse(nn.Module):
    '''
    SRNN dense to sparse module
    '''
    def __init__(self, in_dim, nb_hidden, out_dim, args):
        super(SRNNDense2sparse, self).__init__()
        self.in_dim = in_dim
        self.nb_hidden = nb_hidden
        self.out_dim = out_dim
        self.args = args
        self.dweight_w = torch.nn.parameter.Parameter(torch.empty((self.in_dim, self.out_dim), device=args['device'], dtype=torch.float, requires_grad=True))
        self.dweight_r = torch.nn.parameter.Parameter(torch.empty((self.nb_hidden, self.out_dim), device=args['device'], dtype=torch.float, requires_grad=True))
        # torch.nn.init.normal_(self.dweight_w, mean=0.0, \
        # std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
        # torch.nn.init.normal_(self.dweight_r, mean=0.0, \
        # std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
        nn.init.xavier_uniform_(self.dweight_w)
        # fr = fully recurrent (all to all)
        if self.args['recurrent_mode'] == 'fr':
            nn.init.orthogonal_(self.dweight_r)
        # sr = self-recurrent (one to one)
        elif self.args['recurrent_mode'] == 'sr':
            # weight_diag = torch.rand(self.nb_hidden) # Norm(0,1)
            # self.dweight_r.data = torch.diag_embed(weight_diag)

            weight_diag = torch.eye(self.nb_hidden)  # eye
            self.dweight_r.data = weight_diag
        else:
            raise Exception(
                'Please check the recurrent mode, for now, \
                only fr(fully recurrent) and sr(self-recurrent) are supported.')
        # rT = self.dweight_r.T
        # rN = torch.inverse(self.dweight_r)
        # print(self.dweight_r.min())
        # print(self.dweight_r.max())

        self.Dense2Sparse_function = Dense2SparseSRNNLayer

    def forward(self, x):
        spk_rec, aout_idx = self.Dense2Sparse_function(x, self.dweight_w,
                                                            self.dweight_r,
                                                            self.nb_hidden,
                                                            self.out_dim,
                                                            self.args)
        return spk_rec, aout_idx

    def get_dweight_r(self):
        return self.dweight_r


class SNNDense2sparse(nn.Module):
    '''
    SNN dense to sparse module
    '''
    def __init__(self, in_dim, nb_hidden, out_dim, args):
        super(SNNDense2sparse, self).__init__()
        self.in_dim = in_dim
        self.nb_hidden = nb_hidden
        self.out_dim = out_dim
        self.args = args
        self.dweight_w = torch.nn.parameter.Parameter(torch.empty((self.in_dim, self.out_dim), device=args['device'], dtype=torch.float, requires_grad=True))
        # torch.nn.init.normal_(self.dweight_w, mean=0.0, \
        # std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
        nn.init.xavier_uniform_(self.dweight_w)

        self.Dense2Sparse_function = Dense2SparseSNNLayer

    def forward(self, x):
        spk_rec, aout_idx = self.Dense2Sparse_function(x, self.dweight_w,
                                                            self.nb_hidden,
                                                            self.out_dim,
                                                            self.args)
        return spk_rec, aout_idx
