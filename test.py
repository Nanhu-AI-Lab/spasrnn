import torch
import pytest
#from SRNN_pytorch import sparse_data_generator_torchvision, current2firing_time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
from time import time
import numpy as np
from torch.cuda import amp
from einops import rearrange
from einops.layers.torch import Rearrange
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('.')
from lib.SRNN.snn_layers import *
from lib.SRNN.srnn_cuda import diegaogao_BiSRNN_fMNIST
import numpy as np
import torch
import sys
sys.argv=['']
del sys
from config import args
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib
import pickle
import numpy as np
from scipy import stats
import seaborn as sns
from plot_utils import plot_error, plt_set, panel_specs, label_panel
import matplotlib.style as style

from matplotlib import cm
HEATMAP = "YlGnBu"

_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def plot_gradients(ax, grad, i, j, vmin, vmax, heatmap=None):
    cmap = "YlGnBu" if heatmap is None else heatmap
    sns.heatmap(grad,
                cmap=cmap,
                cbar=False,
                xticklabels=i==1,
                yticklabels=j==0,
                vmin=vmin,
                vmax=vmax,
                ax=ax)

# CUDA configuration
if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is not available')
if args.R_local == 'cuda':
    from lib.SRNN.srnn_cuda import *
else:
    from lib.record.srnn_tor import *

def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    idx = x < thr
    x = np.clip(x, thr + epsilon, 1e9) #make spiking in [thr+epsilon, 1e9]
    T = tau * np.log(x / (x - thr))
    T[idx] = tmax
    return T

def sparse_data_generator_torchvision(X, y, prs, shuffle=True):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """
    batch_size = prs.b
    time_step = prs.time_step
    nb_steps = prs.nb_steps
    nb_inputs = prs.nb_inputs
    device = prs.device

    labels_ = np.array(y, dtype=int)
    number_of_batches = len(X) // batch_size
    number_of_batches = 1 #lh
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = prs.tau_eff if prs.tau_eff is not None else 20e-3
    tau_eff /= time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=int)
    unit_numbers = np.arange(nb_inputs)

    if shuffle:
        if prs.set_random_state == False:
            r = np.random.RandomState(prs.seed)
            r.shuffle(sample_index)
        else:
            np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            c = firing_times[idx] < nb_steps
            times, inputs = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(inputs)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_inputs])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device).long()

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1

def test_DSRNN_SRNN_layer_output():
    #1.参数初始化
    T = 784
    weight_1 = torch.empty((T, args.nb_hidden), device=args.device, dtype=args.dtype,
                requires_grad=True)
    torch.nn.init.normal_(weight_1, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
    
    weight_R = torch.nn.parameter.Parameter(torch.empty((args.nb_hidden, args.nb_hidden), device=args.device, dtype=args.dtype,
                requires_grad=True))
    torch.nn.init.normal_(weight_R, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))

    #2.实例化模型
    net = diegaogao_BiSRNN_fMNIST(T, args ).to(device)
    net_SRNN = BiSRNN_fMNIST(T, args ).to(device)

    #2.1 相同 W R
    net.srnn_cell_1.weight_W.data = weight_1.data
    net_SRNN.weight_1 = weight_1
    net.srnn_cell_1.weight_R = weight_R
    net_SRNN.rnn.srnn_fw.recurrent.weight = weight_R

    #3.相同 input
    data_path = '/home/ubuntu/data/datasets/torch/fashion-mnist' #[To change] Datapath for MNIST
    transform = torchvision.transforms.Compose(
             [torchvision.transforms.ToTensor(),
              torchvision.transforms.Lambda(lambda x: x.view(T, -1))
             ])
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=None, target_transform=None,
                                                          download=True)
    x_train = np.array(train_dataset.data, dtype=float)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    y_train = np.array(train_dataset.targets, dtype=int)

    for x_local, _ in sparse_data_generator_torchvision(x_train, y_train, args):            
        x_local_dense = x_local.to_dense()
        x_local_dense[x_local_dense[:] > 1.] = 1.

    #4.output 
    out_SDRNN = net(x_local_dense)
    out_SRNN  = net_SRNN(x_local_dense)

    #5.assert
    # assert (net.srnn_cell_1.weight_W.data == weight_1.data).all()
    # assert (net_SRNN.weight_1.data == weight_1.data).all()
    # assert (net.srnn_cell_1.weight_R.data == weight_R.data).all()
    # assert (net_SRNN.rnn.srnn_fw.recurrent.weight == weight_R).all()
        
    # assert (out_SDRNN.shape == out_SRNN.shape)    
    #assert torch.allclose(out_SDRNN, out_SRNN)
    # assert ((out_SDRNN - out_SRNN)**2 < 1e-2).all()  #np.array.any()是或操作，任意一个元素为True，输出为True。
    print(out_SDRNN)
    print(out_SRNN)

def test_DSRNN_SRNN_layer_Loss():
        #1.参数初始化
    T = 784
    weight_1 = torch.empty((T, args.nb_hidden), device=args.device, dtype=args.dtype,
                requires_grad=True)
    torch.nn.init.normal_(weight_1, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
    
    weight_R = torch.nn.parameter.Parameter(torch.empty((args.nb_hidden, args.nb_hidden), device=args.device, dtype=args.dtype,
                requires_grad=True))
    torch.nn.init.normal_(weight_R, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))

    #2.实例化模型
    net = diegaogao_BiSRNN_fMNIST(T, args ).to(device)
    net_SRNN = BiSRNN_fMNIST(T, args ).to(device)

    #2.1 相同 W R
    net.srnn_cell_1.weight_W.data = weight_1.data
    net_SRNN.weight_1 = weight_1
    net.srnn_cell_1.weight_R = weight_R
    net_SRNN.rnn.srnn_fw.recurrent.weight = weight_R

    #3.相同 input
    data_path = '/home/ubuntu/data/datasets/torch/fashion-mnist' #[To change] Datapath for MNIST
    transform = torchvision.transforms.Compose(
             [torchvision.transforms.ToTensor(),
              torchvision.transforms.Lambda(lambda x: x.view(T, -1))
             ])
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=None, target_transform=None,
                                                          download=True)
    x_train = np.array(train_dataset.data, dtype=float)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    y_train = np.array(train_dataset.targets, dtype=int)

    for x_local, y_local in sparse_data_generator_torchvision(x_train, y_train, args):            
        x_local_dense = x_local.to_dense()
        x_local_dense[x_local_dense[:] > 1.] = 1.

    #4.output 
    out_SDRNN = net(x_local_dense)
    out_SRNN  = net_SRNN(x_local_dense)
    
    loss_fn = torch.nn.NLLLoss()
    log_softmax_fn = torch.nn.LogSoftmax(dim=1)

    Readout = SNNReadout_Module(args)   
    out_SDRNN = Readout(out_SDRNN)
    out_SDRNN = log_softmax_fn(torch.max(out_SDRNN, 1)[0])
    loss_SDRNN = loss_fn(out_SDRNN, y_local)

    out_SRNN = Readout(out_SRNN)
    out_SRNN = log_softmax_fn(torch.max(out_SRNN, 1)[0])
    loss_SRNN = loss_fn(out_SRNN, y_local)
    #5.assert        
    assert (loss_SDRNN.shape == loss_SRNN.shape)
    assert (loss_SDRNN != 0).all()
    assert (loss_SRNN != 0).all()
    assert ((loss_SDRNN - loss_SRNN)**2 < 1e-2).all()
    assert (loss_SDRNN == loss_SRNN).all()
    #assert torch.allclose(loss_SDRNN, loss_SRNN)

def test_DSRNN_SRNN_backward():
        #1.参数初始化
    T = 784
    weight_1 = torch.empty((T, args.nb_hidden), device=args.device, dtype=args.dtype,
                requires_grad=True)
    torch.nn.init.normal_(weight_1, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
    
    weight_R = torch.nn.parameter.Parameter(torch.empty((args.nb_hidden, args.nb_hidden), device=args.device, dtype=args.dtype,
                requires_grad=True))
    torch.nn.init.normal_(weight_R, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))

    #2.实例化模型
    net = diegaogao_BiSRNN_fMNIST(T, args ).to(device)
    net_SRNN = BiSRNN_fMNIST(T, args ).to(device)

    #2.1 相同 W R
    net.srnn_cell_1.weight_W.data = weight_1.data
    net_SRNN.weight_1 = weight_1
    net.srnn_cell_1.weight_R = weight_R
    net_SRNN.rnn.srnn_fw.recurrent.weight = weight_R

    #3.相同 input
    data_path = '/home/ubuntu/data/datasets/torch/fashion-mnist' #[To change] Datapath for MNIST
    transform = torchvision.transforms.Compose(
             [torchvision.transforms.ToTensor(),
              torchvision.transforms.Lambda(lambda x: x.view(T, -1))
             ])
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=None, target_transform=None,
                                                          download=True)
    x_train = np.array(train_dataset.data, dtype=float)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    y_train = np.array(train_dataset.targets, dtype=int)

    for x_local, y_local in sparse_data_generator_torchvision(x_train, y_train, args):            
        x_local_dense = x_local.to_dense()
        x_local_dense[x_local_dense[:] > 1.] = 1.

    #4.output 
    out_SDRNN = net(x_local_dense)
    out_SRNN  = net_SRNN(x_local_dense)
    
    loss_fn = torch.nn.NLLLoss()
    log_softmax_fn = torch.nn.LogSoftmax(dim=1)

    Readout = SNNReadout_Module(args)   
    out_SDRNN = Readout(out_SDRNN)
    out_SDRNN = log_softmax_fn(torch.max(out_SDRNN, 1)[0])
    loss_SDRNN = loss_fn(out_SDRNN, y_local)

    out_SRNN = Readout(out_SRNN)
    out_SRNN = log_softmax_fn(torch.max(out_SRNN, 1)[0])
    loss_SRNN = loss_fn(out_SRNN, y_local)

    loss_SDRNN.backward()
    loss_SRNN.backward()

    #5.assert  zhiyou yiceng
    assert (( net.srnn_cell_1.weight_W.grad - net_SRNN.weight_1.grad)**2 < 1e-2).all() #权重需要检查 待讨论 .rnn.srnn_fw.dense.weight
    assert (( net.srnn_cell_1.weight_R.grad - net_SRNN.rnn.srnn_bw.recurrent.weight.grad)**2 < 1e-2).all()
    assert ( ( net.srnn_cell_1.weight_W.grad - net_SRNN.weight_1.grad)**2 < 1e-2).all() #权重需要检查 待讨论 .rnn.srnn_fw.dense.weight
    assert (( net.srnn_cell_1.weight_R.grad - net_SRNN.rnn.srnn_bw.recurrent.weight.grad)**2 < 1e-2).all()


T = 784
weight_1 = torch.empty((T, args.nb_hidden), device=args.device, dtype=args.dtype,
            requires_grad=True)
torch.nn.init.normal_(weight_1, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
torch.nn.init.zeros_(weight_1)

weight_R = torch.nn.parameter.Parameter(torch.empty((args.nb_hidden, args.nb_hidden), device=args.device, dtype=args.dtype,
            requires_grad=True))
torch.nn.init.normal_(weight_R, mean=0.0, std=20. * (1.0 - float(np.exp(-0.1))) / np.sqrt(args.nb_hidden))
torch.nn.init.zeros_(weight_R)
print(f"weight_1 size is {weight_1.size()}")
#2.1 相同 W R


#3.相同 input
data_path = '/home/ubuntu/data/datasets/torch/fashion-mnist' #[To change] Datapath for MNIST
transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(T, -1))
            ])
train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=None, target_transform=None,
                                                        download=True)
x_train = np.array(train_dataset.data, dtype=float)
x_train = x_train.reshape(x_train.shape[0], -1) / 255
y_train = np.array(train_dataset.targets, dtype=int)

loss_fn = torch.nn.NLLLoss()
log_softmax_fn = torch.nn.LogSoftmax(dim=1)
Readout = SNNReadout_Module(args)   
Readout_SRNN = SNNReadout_Module(args)   


for x_local, y_local in sparse_data_generator_torchvision(x_train, y_train, args):            
    x_local_dense = x_local.to_dense()
    x_local_dense[x_local_dense[:] > 1.] = 1.

range_w = 20
range_s = range_w
def get_grad( net, flag ):
    if flag == 'diegaogao':
        net.srnn_cell_1.weight_W.data = weight_1.data.clone().detach()
        net.srnn_cell_1.weight_R.data = weight_R.data.clone().detach()
    elif flag == 'srnn':         
        net.rnn.srnn_fw.dense.weight.data = weight_1.T.data.clone().detach()    
        net.rnn.srnn_fw.recurrent.weight.data = weight_R.T.data.clone().detach()

    out = net(x_local_dense)
    #print(out)

    out = Readout(out)
    out = log_softmax_fn(torch.max(out, 1)[0])
    loss = loss_fn(out, y_local)
    loss.backward()

    if flag == 'diegaogao':
        grad_r = net.srnn_cell_1.weight_R.grad.detach().cpu().numpy()[:range_w, :range_w]
        grad_w = net.srnn_cell_1.weight_W.grad.detach().cpu().numpy()[:range_w, :range_w]
    elif flag == 'srnn':    
        grad_r = net.rnn.srnn_fw.recurrent.weight.grad.detach().cpu().numpy()[:range_w, :range_w]
        grad_w = net.rnn.srnn_fw.dense.weight.grad.detach().cpu().numpy()[:range_w, :range_w]
    return grad_w.T, grad_r.T

if __name__ == "__main__":
   
    net = diegaogao_BiSRNN_fMNIST(T, args ).to(device)  
    grad_w_s3gd , grad_r_s3gd = get_grad(net,'diegaogao' )
    print(f"weight_1 size is {weight_1.size()}")
    net_SRNN = BiSRNN_fMNIST(T, args ).to(device)
    grad_w_orig , grad_r_orig = get_grad(net_SRNN,'srnn' )
    #plot
    layout = '''
        A
        '''
    fig = plt.figure(figsize=(12, 8), dpi=150)
    specs, gs = panel_specs(layout, fig=fig)
    N = 2
    M = 3

    vmin_w = np.mean(np.stack((grad_w_orig, grad_w_s3gd)).flatten()) - 1*np.std(np.stack((grad_w_orig, grad_w_s3gd)).flatten())
    vmax_w = np.mean(np.stack((grad_w_orig, grad_w_s3gd)).flatten()) + 1*np.std(np.stack((grad_w_orig, grad_w_s3gd)).flatten())
    vmin_s = min(np.min(grad_r_orig), np.min(grad_r_s3gd))
    vmax_s = max(np.max(grad_r_orig), np.max(grad_r_s3gd))
    max_s = max(abs(vmin_s), abs(vmax_s))
    vmin_s, vmax_s = -max_s, max_s

    triaxes = {}
    subgs = specs['A'].subgridspec(N, M, wspace=0.2, hspace=0.2)
    subgs.set_width_ratios([20, 20, 1])
    for i in range(N):
        for j in range(M):
            triaxes[i, j] = ax = fig.add_subplot(subgs[i, j])
            ax.set_facecolor('lightgrey')
            if i==0:
                plt.fill_between([0, range_w], [range_w, range_w], color="none", hatch="////", edgecolor="silver", linewidth=0.)  # Hatch
                if j==0:
                    grad_w_orig[grad_w_orig==0.] = np.NaN
                    hm = plot_gradients(ax, grad_w_orig, i, j, vmin_w, vmax_w, heatmap=HEATMAP)
                    ax.set_title(r'Original $\nabla W$')
                    ax.set_ylabel('Output Neuron index')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                    ax.set_yticklabels(['0', '5', '10', '15', '20'])
                elif j==1:
                    grad_w_s3gd[grad_w_s3gd==0.] = np.NaN
                    hm = plot_gradients(ax, grad_w_s3gd, i, j, vmin_w, vmax_w, heatmap=HEATMAP)
                    ax.set_title(r'CUDA W GRAD $\nabla W$')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                else:
                    norm = matplotlib.colors.Normalize(vmin=vmin_w, vmax=vmax_w)
                    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=matplotlib.cm.get_cmap(name=HEATMAP), norm=norm, orientation='vertical')
                    cb.ax.tick_params(labelsize=8)
                    cb.ax.yaxis.get_offset_text().set_fontsize(8)
            else:
                plt.fill_between([0, range_s], [range_s, range_s], color="none", hatch="////", edgecolor='silver', linewidth=0.)  # Hatch
                if j==0:
                    grad_r_orig[grad_r_orig==0.] = np.NaN
                    hm = plot_gradients(ax, grad_r_orig, i, j, vmin_s, vmax_s, heatmap=HEATMAP)
                    ax.set_title(r'Original $\nabla R$')
                    ax.set_ylabel('Time index')
                    ax.set_xlabel('Input Neuron index')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                    ax.set_xticklabels(['0', '5', '10', '15', '20'], rotation=0)
                    ax.set_yticklabels(['0', '5', '10', '15', '20'])
                elif j==1:
                    grad_r_s3gd[grad_r_s3gd==0.] = np.NaN
                    hm = plot_gradients(ax, grad_r_s3gd, i, j, vmin_s, vmax_s, heatmap=HEATMAP)
                    ax.set_title(r'CUDA R GRAD $\nabla R$')
                    ax.set_xlabel('Input Neuron index')
                    ax.set_xticks([0, 5, 10, 15, 20])
                    ax.set_yticks([0, 5, 10, 15, 20])
                    ax.set_xticklabels(['0', '5', '10', '15', '20'], rotation=0)
                else:
                    norm = matplotlib.colors.Normalize(vmin=vmin_s, vmax=vmax_s)
                    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=matplotlib.cm.get_cmap(name=HEATMAP), norm=norm, orientation='vertical')
                    cb.ax.tick_params(labelsize=8)
                    cb.ax.yaxis.get_offset_text().set_fontsize(8)
    plt.tight_layout()

    
    plt.savefig('./panel1.png')
    print('OK!')