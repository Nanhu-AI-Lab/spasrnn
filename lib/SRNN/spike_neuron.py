'''
spike neuron
'''
import numpy as np
import torch
import torch.nn.functional as F

surrograte_type = 'sparse'
print('gradient type: ', surrograte_type)

gamma = 0.8
lens = 0.5
r_m = 1
b_j0_value = 1.


# beta_value = 0.184

# beta_value = 1.8
# b_j0_value = .1

# beta_value = .2#1.8
# b_j0_value = .1

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) *(x - mu)) *0.5 / (sigma*sigma)) *\
                        0.3989422804014327 / sigma

class ActFunadp(torch.autograd.Function):
    '''
    define approximate firing function
    '''
    scale = 100.0
    @staticmethod
    def forward(ctx, v_input):  # v_input = membrane potential- threshold
        ctx.save_for_backward(v_input)
        return v_input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        b_input, = ctx.saved_tensors
        grad_input = grad_output
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(b_input*b_input)*2)*0.7978845608028654
        elif surrograte_type == 'MG':
            temp = gaussian(b_input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(b_input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(b_input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-b_input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*b_input.abs())
        elif surrograte_type == 'sparse':
            temp = grad_input / (ActFunadp.scale * \
                    torch.abs(b_input) + 1.0) ** 2
        return temp


act_fun_adp = ActFunadp.apply

def mem_update_adp(inputs, mem, spike, b):

    # alpha = torch.exp(-1. * dt / tau_m).to(device)
    # ro = torch.exp(-1. * dt / tau_adp).to(device)
    # print("a", alpha.device, ro.device)

    # alpha ro chuan jinlai

    beta = 0.

    # b = ro * b + (1 - ro) * spike
    b_value = b_j0_value + beta * b

    alpha = torch.tensor(np.exp(-1e-3/10e-3))# .to(device=device)

    mem = mem * alpha + inputs - b_value * spike
    inputs_ = mem - b_value
    # print("mem", inputs_.device, mem.device, spike.device, b_value.device)
    # spike = F.relu(inputs_)
    spike = act_fun_adp(inputs_)
    return mem, spike, b_value, b


def output_neuron(inputs, mem, tau_m, dt=1, device=None):
    """
    The read out neuron is leaky integrator without spike
    """
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    # chuan alpha jinlai
    mem = mem *alpha +  (1-alpha)*inputs
    return mem
