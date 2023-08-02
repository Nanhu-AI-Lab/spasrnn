from statistics import mean
from click import echo
from nbformat import write
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
from time import time
import numpy as np
from torch.cuda import amp
from config import args

from einops import rearrange
from einops.layers.torch import Rearrange

from lib.SRNN.snn_layers import *

_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


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
def main():
    T = 784
    print(args)
    if args.model_type == 'SRNN':
        net = BiSRNN_fMNIST(T, args)
    elif args.model_type == 'LSTM':
        net = BiLSTM_sMNIST(T)
    elif args.model_type == 'DSRNN':
        net = diegaogao_BiSRNN_fMNIST(T, args)
    Readout = SNNReadout_Module(args)
    print(net)
    net.to(device)
    # print([i for i in net.parameters()])

    optimizer = None
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    data_path = '/home/ubuntu/data/datasets/torch/fashion-mnist' #[To change] Datapath for MNIST
    transform = torchvision.transforms.Compose(
             [torchvision.transforms.ToTensor(),
              torchvision.transforms.Lambda(lambda x: x.view(T, -1))
             ])
    # train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=4)

    # test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transform)
    # testloader = torch.utils.data.DataLoader(test_set, batch_size=args.b, shuffle=False, num_workers=4)

    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=None, target_transform=None,
                                                          download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=None, target_transform=None,
                                                         download=True)

    x_train = np.array(train_dataset.data, dtype=float)
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = np.array(test_dataset.data, dtype=float)
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    y_train = np.array(train_dataset.targets, dtype=int)
    y_test = np.array(test_dataset.targets, dtype=int)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'{args.model}_T_{args.T}_b_{args.b}_{args.opt}_lr_{args.lr}_')
    if args.lr_scheduler == 'CosALR':
        out_dir += f'CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'
    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'fmnist_logs'), purge_step=start_epoch)
    total_time = 0

    loss_fn = torch.nn.NLLLoss()
    log_softmax_fn = torch.nn.LogSoftmax(dim=1)
    
    backward_mean = []
    forward_mean = []

    for epoch in range(start_epoch, args.epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        backward_time = 0
        time_i = 0
        forward_time = 0

        for x_local, y_local in sparse_data_generator_torchvision(x_train, y_train, args):            
            optimizer.zero_grad()
            x_local_dense = x_local.to_dense()
            x_local_dense[x_local_dense[:] > 1.] = 1.
            if args.amp:
                with amp.autocast():
                    out_fr = net(x_local_dense)
                    loss = loss_fn(out_fr, y_local)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                start_time = time.time()
                out_net = net(x_local_dense)
                train_end_time = time.time()

                out_fr = Readout(out_net)
                out_fr = log_softmax_fn(torch.max(out_fr, 1)[0])
                loss = loss_fn(out_fr, y_local)
                loss.backward()
                back_time = time.time()
                backward_time = backward_time+back_time-train_end_time
                forward_time = forward_time + train_end_time-start_time
                # print(f"forward {forward_time}, backward {backward_time}")
                optimizer.step()
            train_samples += y_local.numel()
            train_loss += loss.item() * y_local.numel()
            train_acc += (out_fr.argmax(1) == y_local).float().sum().item()
            time_i += 1
        backward_time = backward_time/time_i
        forward_time = forward_time/time_i
        train_loss /= train_samples
        train_acc /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()
        train_end_time = time.time()
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for x_local, y_local in sparse_data_generator_torchvision(x_test, y_test, args):    
                x_local_dense = x_local.to_dense()
                x_local_dense[x_local_dense[:] > 1.] = 1.        
                out_net = net(x_local_dense)
                out_fr = Readout(out_net)

                out_fr = log_softmax_fn(torch.max(out_fr, 1)[0])
                loss = loss_fn(out_fr, y_local)
                
                test_samples += y_local.numel()
                test_loss += loss.item() * y_local.numel()
                test_acc += (out_fr.argmax(1) == y_local).float().sum().item()
        test_end_time = time.time()

        test_loss /= test_samples
        test_acc /= test_samples
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch+1)
            writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        # torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        # print(args)
        # print(out_dir)
        forward_mean.append(forward_time)
        backward_mean.append(backward_time)

        print(f'epoch={epoch+1}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}')
        print(f'forward time={mean(forward_mean)}, back_time={mean(backward_mean)}')


if __name__ == '__main__':
    main()
