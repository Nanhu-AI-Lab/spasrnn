from cProfile import label
from curses.ascii import SP
from locale import D_FMT
from matplotlib import projections
import torch
import torch.nn.functional as F
import numpy as np
import s3gd_cuda
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

cuda_dense = s3gd_cuda.s3gd_s_backward_Dense2Sparse
cuda_sparse = s3gd_cuda.s3gd_s_backward_master

dense_spk_trace=torch.tensor(np.load("./test_epoch/dense_spk_trace.npy"))
dense_spk_trace_R=torch.tensor(np.load("./test_epoch/dense_spk_trace_R.npy"))
sparse_spk_trace_R=torch.tensor(np.load("./test_epoch/sparse_spk_trace_R.npy"))
sparse_spk_trace=torch.tensor(np.load("./test_epoch/sparse_spk_trace.npy"))


dense_grad_output = torch.tensor(np.load("./test_epoch/dense_grad_output.npy"))
sparse_grad_output = torch.tensor(np.load("./test_epoch/sparse_grad_output.npy"))
dense_grad_weights = torch.tensor(np.load("./test_epoch/dense_grad_weights.npy"))
sparse_grad_weights = torch.tensor(np.load("./test_epoch/sparse_grad_weights.npy"))
dense_spk_trace = torch.tensor(np.load("./test_epoch/dense_spk_trace.npy"))
sparse_spk_trace = torch.tensor(np.load("./test_epoch/sparse_spk_trace.npy"))


def init_Data():
    ds_out=torch.tensor(np.load("./test_new/ds_out.npy"), device='cuda')
    aout_b=torch.tensor(np.load("./test_new/aout_b.npy"), device='cuda')
    aout_t=torch.tensor(np.load("./test_new/aout_t.npy"), device='cuda')
    aout_i=torch.tensor(np.load("./test_new/aout_i.npy"), device='cuda')
    alphas=torch.tensor(np.load("./test_new/alphas.npy"), device='cuda')
    weight=torch.tensor(np.load("./test_new/weight.npy"), device='cuda')
    # Constants
    total_num_threads = 320864
    batch_size = 256
    nb_steps = 100
    den2spa_output = 200
    nb_hidden = 200
    T_window = 92
    return_list = [ds_out, aout_b, aout_t, aout_i, weight, alphas, total_num_threads, batch_size, nb_steps, den2spa_output,nb_hidden, T_window]
    return return_list

def Dense_Output(ds_out,aout_b,aout_t,aout_i,weight, alphas,
    total_num_threads, batch_size, nb_steps, den2spa_output, nb_hidden, T_window):
    
    bjt = aout_b * nb_hidden * nb_steps + aout_i * nb_steps + aout_t
    bjt, bjt_idx = torch.sort(bjt)
    ds_out = ds_out[bjt_idx]


    # Find indices to start from and frequencies (output)
    b = torch.div(bjt, (nb_hidden * nb_steps), rounding_mode='trunc')
    j = torch.div((bjt - b * nb_hidden * nb_steps), nb_steps, rounding_mode='trunc')
    ts_out = bjt - b * nb_hidden * nb_steps - j * nb_steps
    bj = b * nb_hidden + j
    aout_bj_freqs = torch.bincount(bj, minlength=batch_size*nb_hidden)  # How many t for a given b and j
    idxs = torch.nonzero(aout_bj_freqs)  # Where to put the results
    ends = torch.cumsum(aout_bj_freqs[idxs], dim=0) - 1  # Indices
    bj_ends = torch.full((batch_size*nb_hidden,), -1, device="cuda")
    bj_ends[idxs] = ends

    #########################
    # total threads
    total_num_threads = bj.numel()
    ###############################################################

    # Call kernel to compute ds gradient
    T_window = int(T_window)
    Dense_grad_output = cuda_dense(
                                    # Computing
                                    ds_out,  # Values to compute deltas (\red{dS[bjt]})
                                    ts_out,
                                    bj_ends,
                                    bj,
                                    alphas,  # Powers of alpha
                                    # Constants
                                    total_num_threads, batch_size, nb_steps, den2spa_output, nb_hidden, T_window
                                    )

    Dense_grad_output_w = torch.einsum("abd,cd->abc",Dense_grad_output, weight)
    return Dense_grad_output, Dense_grad_output_w
def sparse(ds_out,aout_b,aout_t,aout_i,weight, alphas,batch_size, nb_hidden, nb_steps):
    spk_mlp = torch.ones([256, 100, 200], device='cuda')
    idx_tmp = torch.nonzero(spk_mlp >0)
    ain_b = idx_tmp[:, 0]
    ain_t = idx_tmp[:, 1]
    ain_i = idx_tmp[:, 2]
    bjt = aout_b * nb_hidden * nb_steps + aout_i * nb_steps + aout_t
    bjt, bjt_idx = torch.sort(bjt)
    ds_out = ds_out[bjt_idx]

    # Find indices to start from and frequencies (output)
    b = torch.div(bjt, (nb_hidden * nb_steps), rounding_mode='trunc')
    j = torch.div((bjt - b * nb_hidden * nb_steps), nb_steps, rounding_mode='trunc')
    ts_out = bjt - b * nb_hidden * nb_steps - j * nb_steps
    bj = b * nb_hidden + j
    aout_bj_freqs = torch.bincount(bj, minlength=batch_size*nb_hidden)  # How many t for a given b and j
    idxs = torch.nonzero(aout_bj_freqs)  # Where to put the results
    ends = torch.cumsum(aout_bj_freqs[idxs], dim=0) - 1  # Indices
    bj_ends = torch.full((batch_size*nb_hidden,), -1, device="cuda")
    bj_ends[idxs] = ends

    # Find indices to start from and frequencies (input)
    bt_input_full = ain_b * nb_steps + ain_t
    bt_in_unique = torch.unique(bt_input_full)
    b = torch.div(bt_in_unique, nb_steps, rounding_mode='trunc')
    ts_in = bt_in_unique - b * nb_steps
    ain_b_freqs = torch.bincount(b, minlength=batch_size)  # Get frequencies of input batch
    idxs = torch.nonzero(ain_b_freqs)  # Where to put the results
    ends = torch.cumsum(ain_b_freqs[idxs], dim=0) - 1  # Indices
    b_ends = torch.full((batch_size,), -1, device="cuda")
    b_ends[idxs] = ends

    #########################
    # Get bj_out_unique
    bj_out_unique = torch.unique(bj)
    total_num_threads = bj_out_unique.numel()

    # Get counts
    b_out_unique = torch.div(bj_out_unique, nb_hidden, rounding_mode='trunc')
    b_in_unique = torch.div(bt_in_unique, nb_steps, rounding_mode='trunc')
    b_out_unique_freqs = torch.bincount(b_out_unique, minlength=batch_size)  # Get frequencies of input batch
    b_in_unique_freqs = torch.bincount(b_in_unique, minlength=batch_size)  # Get frequencies of input batch

    # Sparse matrix size
    bM_freqs = torch.cumsum(b_out_unique_freqs * b_in_unique_freqs, 0)
    M = bM_freqs[-1]
    bM_freqs = (bM_freqs).roll(1)
    bM_freqs[0] = 0

    # Get starts for each bj
    bM_starts = torch.roll(bM_freqs, 1)
    bM_starts[0] = 0  # Start for batch
    bj_out_unique_freqs = torch.bincount(bj_out_unique, minlength=batch_size * nb_hidden)  # Get frequencies of each unique bj

    bj_out_unique_freqs = bj_out_unique_freqs.reshape(batch_size, -1).cumsum(1).roll(1)
    bj_out_unique_freqs[:, 0] = 0
    bj_out_unique_freqs = bj_out_unique_freqs.reshape(-1)  # Flatten
    ###############################################################


    # Call kernel to compute ds gradient
    ain_bt_freqs = torch.bincount(bt_input_full, minlength=batch_size*nb_steps)  # Get frequencies of input batch #256*100
    ain_bt_starts = torch.cumsum(ain_bt_freqs, 0).roll(1)
    ain_bt_starts[0] = 0
    grad_input = cuda_sparse(
                            # Computing
                            ds_out,  # Values to compute deltas (\red{dS[bjt]})
                            ts_out,  # Times active output values to compute deltas
                            bj_ends,  # Index fot ts_out with last time to read. Points to last time for each bj
                            aout_bj_freqs,  # How many t we need to compute for a given b&j
                            # Recording
                            ts_in,  # Times to record
                            b_ends,  # Index  for ts_in with last time to record each batch
                            ain_b_freqs,  # How many t we need to record for a given batch (for tensor creation)
                            # Other
                            alphas,  # Powers of alpha
                            weight,
                            # Writing deltas
                            b_in_unique_freqs,
                            bj_out_unique_freqs,
                            bj_out_unique,
                            # Computing gradient
                            ain_bt_freqs,
                            ain_bt_starts,
                            ain_i,
                            # Constants
                            total_num_threads, batch_size, nb_steps, 200, nb_hidden
                                    )
    return grad_input


if __name__ == "__main__":
    data_list = init_Data()
    Dense_CUDA, Dense_CUDA_w = Dense_Output(data_list[0], data_list[1], data_list[2], data_list[3], 
                    data_list[4], data_list[5], data_list[6],data_list[7],
                    data_list[8], data_list[9], data_list[10],data_list[11])
    Sparse_CUDA = sparse(data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5], data_list[7], data_list[10], data_list[8])
    
    Dense_CUDA = Dense_CUDA.detach().cpu().numpy()
    Dense_CUDA_w = Dense_CUDA_w.detach().cpu().numpy()
    Sparse_CUDA = Sparse_CUDA.detach().cpu().numpy()

    bb = 150
    NN = 100

    # Dense_data_bd = Dense_CUDA[bb,:,NN]
    Dense_data = Dense_CUDA_w[bb,:,NN]
    Sparse_Data = Sparse_CUDA[bb,:,NN]

    # Dense_data = dense_grad_weights[bb,:]
    # Sparse_Data = sparse_grad_weights[bb,:]

    # Dense_data = Dense_CUDA_w[bb,:,NN]
    # Sparse_Data = sparse_spk_trace[bb,:,NN]

    plt.figure()
    plt.grid(True)
    # plt.plot(Dense_data_bd, label='Dense2sparse_before')
    # plt.legend()
    plt.plot(Dense_data, label='Dense2sparse')
    plt.legend()
    plt.plot(Sparse_Data, label='Sparse')
    plt.legend()
    plt.savefig("./pic/NNNNNNew-"+str(bb)+"-"+str(NN)+".png")

    None