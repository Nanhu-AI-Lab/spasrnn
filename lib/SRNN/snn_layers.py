'''
snn layers

Author: Wang Kun
'''
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
# from .spike_neuron import *
from .spike_neuron import b_j0_value, mem_update_adp


b_j0 = b_j0_value
# torch.backends.cudnn.enabled = False


def SNNReadout(spk_rec, weight, prs):
    batch_size =prs['batch_size']
    nb_steps =prs['nb_steps']
    nb_outputs = prs['nb_outputs']

    device ='cuda' # prs['device']
    dtype = torch.float # prs['dtype']

    h2 = torch.einsum('abc,cd->abd', (spk_rec, weight))
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps - 1):
        # new_out = prs['beta_readout'] * out + \
        # (1-prs['beta_readout'])*h2[:, t, :]
        new_out = float(np.exp(-0.1)) * out + \
            (1-float(np.exp(-0.1)))*h2[:, t, :]

        out = new_out
        out_rec.append(out)
    out_rec = torch.stack(out_rec, dim=1)
    return out_rec

# Bidirectional SRNN for sequential processing
class BiSRNNFMNIST(nn.Module):
    '''
    Bidirectional SRNN for sequential processing
    '''
    def __init__(self, dim, args):
        super(BiSRNNFMNIST).__init__()

        self.dim = dim #input dimention of every timestep
        self.rnn = SBiRNN(784, 200, args) # Bidirectional SRNN
        self.weight_1 = torch.empty((self.dim, args.nb_hidden),
                                     device=args.device, dtype=args.dtype,
                                     requires_grad=True)
        torch.nn.init.normal_(self.weight_1, mean=0.0,
                              std=20. * (1.0 - float(np.exp(-0.1))) / \
                                np.sqrt(args.nb_hidden))
        self.weight_2 = torch.empty((args.nb_hidden*2, args.nb_hidden),
                                     device=args.device, dtype=args.dtype,
                                     requires_grad=True)
        torch.nn.init.normal_(self.weight_2, mean=0.0,
                              std=20. * (1.0 - float(np.exp(-0.1))) / \
                                np.sqrt(args.nb_hidden))
        self.weight_3 = torch.empty((args.nb_hidden, args.nb_outputs),
                                     device=args.device, dtype=args.dtype,
                                     requires_grad=True)
        torch.nn.init.normal_(self.weight_3, mean=0.0,\
                              std=20. * (1.0 - float(np.exp(-0.1))) / \
                                np.sqrt(args.nb_hidden))


    def forward(self, x):
        # x.shape = [N, C, H, W]
        # x.squeeze_(1)  # [N, H, W]
        # x = x.permute(2, 0, 1)  # [W, N, H]
        # Batch,Time,Input x Input,Output -> Batch,Time,Output
        x = torch.einsum('abc,cd->abd', (x, self.weight_1))
        x = self.rnn(x)
        # x = self.fc(x)
        # Batch,Time,Input x Input,Output -> Batch,Time,Output
        # x = torch.einsum('abc,cd->abd', (x, self.weight_2))
        # x = SNNReadout(x, self.weight_3)
        return x

# Spiking Sequencer model (https://arxiv.org/pdf/2205.01972)
class BiSRNN2D(nn.Module):
    '''
    Spiking Sequencer model (https://arxiv.org/pdf/2205.01972)
    '''
    def __init__(self, dim):
        super(BiSRNN2D).__init__()

        self.dim = dim
        self.rnn_v = SBiRNN(dim, dim//4) #spiking RNN for vertical mixing
        self.rnn_h = SBiRNN(dim, dim//4) #spiking RNN for horizontal mixing
        # self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        batch, height, width, channel = x.shape

        v = self.rnn_v(x.permute(0, 2, 1, 3).reshape(-1, height, channel))
        v = v.reshape(batch, width, height, -1).permute(0, 2, 1, 3)
        h = self.rnn_h(x.reshape(-1, width, channel))
        h = h.reshape(batch, height, width, -1)
        x = torch.cat([v, h], dim=-1)
        # x = self.fc(x)

        return x

#[For comparison] Bidirectional LSTM with same scale
class BiLSTMSMNIST(nn.Module):
    '''
    [For comparison] Bidirectional LSTM with same scale
    '''
    def __init__(self, dim):
        super(BiLSTMSMNIST).__init__()

        self.dim = dim
        self.linear = nn.Linear(dim, 200)
        self.birnn = nn.LSTM(200, 200, bidirectional=True)
        self.fc = nn.Linear(200 * 2, 10)
        # self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x.squeeze_(1)  # [N, H, W]
        x = self.linear(x)
        x = self.birnn(x)[0]
        x = self.fc(x)
        return x.mean(1)

# Bidirectional SRNN
class SBiRNN(nn.Module):
    '''
    Bidirectional SRNN
    '''
    def __init__(self, input_dim, output_dim, args):
        super(SBiRNN).__init__()

        self.srnn_fw = SRNN(input_dim, output_dim)

        self.srnn_bw = SRNN(input_dim, output_dim)
        if args.srnn_out_choose == 'spike':
            self.srnn_out_spike = True
        else:
            self.srnn_out_spike = False

    def forward(self, input_spike_train):
        batch, time, _ = input_spike_train.shape

        self.srnn_fw.init_neuron_state(batch)
        self.srnn_bw.init_neuron_state(batch)

        output_spike_train_fw = []
        output_spike_train_bw = []

        # self.srnn_fw.alpha = torch.exp(-1.*self.srnn_fw.dt/self.srnn_fw.tau_m)
        # self.srnn_fw.ro = torch.exp(-1.*self.srnn_fw.dt/self.srnn_fw.tau_adp)

        # self.srnn_bw.alpha = torch.exp(-1.*self.srnn_bw.dt/self.srnn_bw.tau_m)
        # self.srnn_bw.ro = torch.exp(-1.*self.srnn_bw.dt/self.srnn_bw.tau_adp)

        for t in range(time):
            input_spike_fw, input_spike_bw = \
                input_spike_train[:,t,:], input_spike_train[:,-t,:]

            if self.srnn_out_spike:
                _, spike_fw = self.srnn_fw(input_spike_fw)
                _, spike_bw = self.srnn_bw(input_spike_bw)
            else:
                spike_fw, _ = self.srnn_fw(input_spike_fw)
                spike_bw, _ = self.srnn_bw(input_spike_bw)
            output_spike_train_fw.append(spike_fw.unsqueeze(0))
            output_spike_train_bw.insert(0, spike_bw.unsqueeze(0))
        output_spike_train_fw = torch.cat(output_spike_train_fw, dim=0).\
            permute(1,0,2)
        output_spike_train_bw = torch.cat(output_spike_train_bw, dim=0).\
            permute(1,0,2)

        output_spike_train_all = torch.cat((output_spike_train_fw,
                                            output_spike_train_bw),
                                            dim=-1) # B*T*2H

        return output_spike_train_all

# SRNN in https://arxiv.org/abs/2103.12593
class SRNN(nn.Module):
    '''
    SRNN in https://arxiv.org/abs/2103.12593
    '''
    def __init__(self, input_dim, output_dim, \
                 is_adaptive=1, device='cuda', bias=False):
        super(SRNN).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device

        self.b_j0 = b_j0
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.recurrent = nn.Linear(output_dim,output_dim,bias=bias)
        print(f'srnn dense size is {self.dense.weight.size()},\
            input_dim={input_dim},output_dim={output_dim}')
        #Parameters initialization
        # nn.init.xavier_uniform_(self.dense.weight)
        # nn.init.xavier_uniform_(self.recurrent.weight)
        torch.nn.init.normal_(self.dense.weight, mean=0.0,
                              std=20. * (1.0 - float(np.exp(-0.1))) / \
                                np.sqrt(self.output_dim))
        torch.nn.init.normal_(self.recurrent.weight, mean=0.0,
                              std=20. * (1.0 - float(np.exp(-0.1))) / \
                                np.sqrt(self.output_dim))


        # nn.init.constant_(self.dense.bias, 0)
        # nn.init.constant_(self.recurrent.bias, 0)


    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.recurrent.weight,
                self.recurrent.bias,self.tau_m,self.tau_adp]

    def init_neuron_state(self, batch_size):
        self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).\
            to(self.device)
        # self.mem = Variable(torch.zeros(batch_size,\
        # self.output_dim)*self.b_j0).\
        # to(self.device)
        self.spike = Variable(torch.zeros(batch_size,self.output_dim)).\
            to(self.device)
        self.b = Variable(torch.ones(batch_size,
                                     self.output_dim)*self.b_j0).\
            to(self.device)

    def forward(self, input_spike):
        d_input = self.dense(input_spike) + self.recurrent(self.spike)

        self.mem,self.spike,_,self.b = mem_update_adp(inputs=d_input,
        mem=self.mem,
        spike=self.spike,
        b=self.b,
        device=self.device,
        isAdapt=self.is_adaptive)

        return self.mem, self.spike

def multi_normal_initilization(param, means=None, stds=None):
    if means is None:
        means = [10,200]
    if stds is None:
        stds=[5,20]
    # init_start_time = time.time()
    shape_list = param.shape
    if len(shape_list) == 1:
        num_total = shape_list[0]
    elif len(shape_list) == 2:
        num_total = shape_list[0]*shape_list[1]

    num_per_group = int(num_total/len(means))
    # if num_total%len(means) != 0:
    num_last_group = num_total%len(means)
    a = []
    for i in range(len(means)):
        a = a+ np.random.normal(means[i],stds[i],size=num_per_group).tolist()
        if i == len(means):
            a = a+ np.random.normal(means[i],
                                    stds[i],
                                    size=num_per_group+num_last_group).tolist()
    p = np.array(a).reshape(shape_list)
    # init_end_time = time.time()
    # print('init time',init_end_time-init_start_time)
    with torch.no_grad():
        param.copy_(torch.from_numpy(p).float())

    return param
