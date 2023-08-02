'''
Dense2SparseSRNNLayer and Dense2SparseSNNLayer
'''
import numpy as np
import torch
#import time
import s3gd_cuda

surr_grad_spike_cuda = s3gd_cuda.surr_grad_spike
s3gd_backward_cuda = s3gd_cuda.s3gd_w_backward_Dense2Sparse
s3gd_s_backward_master = s3gd_cuda.s3gd_s_backward_Dense2Sparse
s3gd_wR_backward_cuda = s3gd_cuda.s3gd_wR_backward_Dense2Sparse
s3gd_w_self_R_backward_cuda = s3gd_cuda.s3gd_w_self_R_backward_Dense2Sparse


class Dense2SparseSRNNLayer(torch.autograd.Function):
    '''
    d2s-sRnn
    '''
    @staticmethod
    def forward(ctx, inputs, weight, weight_r, nb_hidden, den2spa_output, prs):
        # weight_r, spk_rec, ctx=context
        batch_size = inputs.shape[0]
        nb_inputs = inputs.shape[2]  # Batch, Time, Units
        nb_steps = inputs.shape[1]
        beta = float(np.exp(-float(prs['time_step']) / float(prs['tau_mem'])))
        th = float(prs['th'])
        b_th = float(prs['b_th'])
        device = prs['device']
        reset_m = prs['reset_m']
        recurrent_m = prs['recurrent_mode']
        ain_idxs = torch.nonzero(inputs < 100.0)
        ain_i = ain_idxs[:, 2]

        t_window = torch.div(torch.log10(torch.tensor(float(prs['alpha_window']))),
                             torch.log10(torch.tensor(beta)),
                             rounding_mode='trunc')

        # Forward
        with torch.no_grad():
            # + weight_bias
            # Batch,Time,Input x Input,Output -> Batch,Time,Output
            h1 = torch.einsum('abc,cd->abd',
                              (inputs, weight))

            mem = torch.zeros((batch_size, den2spa_output),
                              device=device, dtype=torch.float)
            r_input = torch.zeros(
                (batch_size, den2spa_output), device=device, dtype=torch.float)

            spk = torch.zeros((batch_size, den2spa_output),
                              device=device, dtype=torch.float)
            out = torch.zeros((batch_size, nb_hidden),
                              device=device, dtype=torch.float)

            trace = torch.zeros((batch_size, nb_inputs),
                                device=device, dtype=torch.float)
            trace_r = torch.zeros(
                (batch_size, den2spa_output), device=device, dtype=torch.float)

            mem_rec = [mem]
            spk_rec = [spk]
            spk_trace = [trace]
            spk_trace_r = [trace_r]
            # Membrane - th
            if reset_m == 'soft':
                for t in range(nb_steps - 1):
                    # if recurrent_m == 'fr': # fully recurrent
                    #     r_input = torch.einsum('ac,cd->ad', (out, weight_r))
                    # elif recurrent_m == 'sr': # self recurrent
                    # only using the diag to make self-recurrence
                    #     weight_r = weight_r * torch.eye(weight_r.size(0), \
                    # device=device, dtype=torch.float)
                    #     r_input = torch.einsum('ac,cd->ad', (out, weight_r))
                    # else:
                    #     raise Exception('Please check the recurrent mode, \
                    #       for now, only fr(fully recurrent) and \
                    #       sr(self-recurrent) are supported.')
                    r_input = torch.einsum('ac,cd->ad', (out, weight_r))
                    mem = beta * mem + h1[:, t, :] - out + r_input

                    mthr = mem - th
                    out = torch.zeros_like(mem)
                    out[mthr > 0] = 1.0
                    # =================== NEW STUFF ======================== #
                    trace = beta * trace + inputs[:, t, :]
                    trace_r = beta * trace_r + out
                    spk_trace.append(trace)
                    spk_trace_r.append(trace_r)
                    # ================================== #
                    mem_rec.append(mem)
                    spk_rec.append(out)

                mem_rec = torch.stack(mem_rec, dim=1)
                spk_rec = torch.stack(spk_rec, dim=1)
                spk_trace = torch.stack(spk_trace, dim=1)
                spk_trace_r = torch.stack(spk_trace_r, dim=1)
                # =================== NEW STUFF ==================== #
                # Indices of mem_rec above threshold mask [b, t, j]
                aout_idxs = torch.nonzero(torch.logical_and(mem_rec > b_th,\
                            mem_rec < 2. * th - b_th))

                aout_mem = (mem_rec[aout_idxs[:, 0],
                            aout_idxs[:, 1], aout_idxs[:, 2]] - th)
            # Membrane set to 0
            elif reset_m == 'hard':
                last_mthr = torch.zeros_like(mem)
                for t in range(nb_steps - 1):
                    mem[last_mthr > 0] = 0.0  # hard reset

                    # if recurrent_m == 'fr': # fully recurrent
                    #     r_input = torch.einsum('ac,cd->ad', (out, weight_r))
                    # elif recurrent_m == 'sr': # self recurrent
                    #only using the diag to make self-recurrence
                    #     weight_r = weight_r * torch.eye(weight_r.size(0), \
                    # device=device, dtype=torch.float)
                    #     r_input = torch.einsum('ac,cd->ad', (out, weight_r))
                    # else:
                    #     raise Exception('Please check the recurrent mode,\
                    #           for now, only fr(fully recurrent) and \
                    # sr(self-recurrent) are supported.')
                    r_input = torch.einsum('ac,cd->ad', (out, weight_r))
                    mem = beta * mem + h1[:, t, :] + r_input

                    mthr = mem - th
                    last_mthr = mthr
                    out = torch.zeros_like(mem)
                    out[mthr > 0] = 1.0
                    # ==================== NEW STUFF ========================= #
                    trace = beta * trace + inputs[:, t, :]
                    trace_r = beta * trace_r + out
                    spk_trace.append(trace)
                    spk_trace_r.append(trace_r)
                    # =============================================== #
                    mem_rec.append(mem)
                    spk_rec.append(out)

                mem_rec = torch.stack(mem_rec, dim=1)
                spk_rec = torch.stack(spk_rec, dim=1)
                spk_trace = torch.stack(spk_trace, dim=1)
                spk_trace_r = torch.stack(spk_trace_r, dim=1)
                # ==================== NEW STUFF =================== #
                 # Indices of mem_rec above threshold mask [b, t, j]
                aout_idxs = torch.nonzero(torch.logical_and(mem_rec > b_th,mem_rec < 2. * th - b_th))

                aout_mem = (mem_rec[aout_idxs[:, 0],
                            aout_idxs[:, 1], aout_idxs[:, 2]] - th)
            else:
                raise Exception(
                    'The reset mode is only supported by hard and soft, \
                        please check it in config.py')
            # Indices out
            aout_b = aout_idxs[:, 0]
            aout_t = aout_idxs[:, 1]
            aout_i = aout_idxs[:, 2]
            # Indices in
            # =============================================== #

        # Save for backward
        alphas = (torch.full((nb_steps,), beta, device=device)
                  ** torch.arange(nb_steps, device=device))

        ctx.save_for_backward(spk_trace,
                              spk_trace_r,
                              aout_b,
                              aout_t,
                              aout_i,
                              ain_i,
                              aout_mem,
                              t_window,
                              alphas,
                              weight,
                              weight_r)
        # ctx.save_for_backward(spk_trace,
        #                       aout_b,
        #                       aout_t,
        #                       aout_i,
        #                       aout_mem,
        #                       ain_b,
        #                       ain_t,
        #                       ain_i,
        #                       alphas,
        #                       weight,
        #                       weight_r)

        ctx.batch_size = batch_size
        ctx.nb_steps = nb_steps
        ctx.nb_hidden = nb_hidden
        ctx.nb_inputs = nb_inputs
        ctx.den2spa_output = den2spa_output
        ctx.recurrent_m = recurrent_m
        spk_rec.requires_grad = True
        aout_idxs.requires_grad = False
        return spk_rec, aout_idxs

    @staticmethod
    def backward(ctx, grad_output, grad_dummy1):
        # ATTENTION! DO NOT delete grad_dummy1, \
        # this pramater is used to balance the grad number of input and output!
        spk_trace, spk_trace_r, aout_b, \
            aout_t, aout_i, _, aout_mem, \
                t_window, alphas, weight, weight_r = ctx.saved_tensors

        batch_size, nb_steps, nb_inputs, \
            nb_hidden, den2spa_output, recurrent_m = \
                ctx.batch_size, ctx.nb_steps, \
                    ctx.nb_inputs, ctx.nb_hidden, \
                        ctx.den2spa_output, ctx.recurrent_m
        dense_grad_output, grad_weights, grad_weights_r = None, None, None

        # Active output values gradient
        grad_output = grad_output[aout_b, aout_t, aout_i]
        grad_output_idxs = torch.nonzero(grad_output, as_tuple=True)[0]
        grad_output = grad_output[grad_output_idxs]
        aout_b = aout_b[grad_output_idxs]
        aout_t = aout_t[grad_output_idxs]
        aout_i = aout_i[grad_output_idxs]
        aout_mem = aout_mem[grad_output_idxs]
        if grad_output.numel() == 0:
            grad_weights = torch.zeros_like(weight)
            grad_weights_r = torch.zeros_like(weight_r)

            dense_grad_output = torch.zeros(
                (batch_size, nb_steps, nb_inputs), device=grad_output.device)
            return dense_grad_output, grad_weights, grad_weights_r, \
                None, None, None
            # return grad_output, grad_weights, None, None, None

        ds_out = surr_grad_spike_cuda(aout_mem, grad_output)  # ds/dv

        torch.cuda.synchronize()
        #weight_start = time.time()

        if ctx.needs_input_grad[2]:
            if recurrent_m == 'fr':
                grad_weights_r = s3gd_wR_backward_cuda(spk_trace_r,
                                                       aout_b,
                                                       aout_t,
                                                       aout_i,
                                                       ds_out,
                                                       nb_hidden,
                                                       nb_hidden)
            elif recurrent_m == 'sr':
                grad_weights_r = s3gd_w_self_R_backward_cuda(spk_trace_r,
                                                             aout_b,
                                                             aout_t,
                                                             aout_i,
                                                             ds_out,
                                                             nb_hidden,
                                                             nb_hidden)
            else:
                raise Exception('Please check the recurrent mode, for now, \
                    only fr(fully recurrent) and \
                        sr(self-recurrent) are supported.')
        torch.cuda.synchronize()
        #weight_end_time = time.time()
        #weight_time = weight_end_time - weight_start
        # Weight gradient
        if ctx.needs_input_grad[1]:
            grad_weights = s3gd_backward_cuda(spk_trace, aout_b, aout_t,
                                              aout_i, ds_out,
                                              nb_inputs, nb_hidden)
        torch.cuda.synchronize()
        #wr_weight_end_time = time.time()
        #wr_time = wr_weight_end_time - weight_end_time
        # Input spikes gradient
        if ctx.needs_input_grad[0]:  # This is false for first layer

            # Reorder active output indices to (b,j,t) format
            bjt = aout_b * nb_hidden * nb_steps + aout_i * nb_steps + aout_t
            bjt, bjt_idx = torch.sort(bjt)
            ds_out = ds_out[bjt_idx]

            # Find indices to start from and frequencies (output)
            b = torch.div(bjt, (nb_hidden * nb_steps), rounding_mode='trunc')
            j = torch.div((bjt - b * nb_hidden * nb_steps),
                          nb_steps, rounding_mode='trunc')
            ts_out = bjt - b * nb_hidden * nb_steps - j * nb_steps
            bj = b * nb_hidden + j
            # How many t for a given b and j
            aout_bj_freqs = torch.bincount(
                bj, minlength=batch_size * nb_hidden)
            idxs = torch.nonzero(aout_bj_freqs)  # Where to put the results
            ends = torch.cumsum(aout_bj_freqs[idxs], dim=0) - 1  # Indices
            bj_ends = torch.full((batch_size * nb_hidden,), -1, device='cuda')
            bj_ends[idxs] = ends

            #########################
            # total threads
            total_num_threads = bj.numel()
            ###############################################################

            # Call kernel to compute ds gradient
            t_window = int(t_window)
            dense_grad_output = s3gd_s_backward_master(
                # Computing
                ds_out,  # Values to compute deltas (\red{dS[bjt]})
                ts_out,
                bj_ends,
                bj,
                alphas,  # Powers of alpha
                # Constants
                total_num_threads,
                batch_size,
                nb_steps,
                den2spa_output,
                nb_hidden,
                t_window
            )

            dense_grad_output = torch.einsum(
                'abd,cd->abc', dense_grad_output, weight)
        torch.cuda.synchronize()
        #s_end_time = time.time()
        #s_time = s_end_time - wr_weight_end_time
        # try:
        #     len(ain_i)
        # except TypeError:
        #     print(f'ain_i is {type(ain_i)}, \
        #           weight grad time is {weight_time}, \
        #           r_grad is {wr_time}, s_grad is {s_time}')
        # else:
        #     print(f'ain_i is {len(ain_i)}, \
        #           weight grad time is {weight_time}, \
        #           r_grad is {wr_time}, s_grad is {s_time}')

        return dense_grad_output, grad_weights, grad_weights_r, None, None, None


Dense2SparseSRNNLayer = Dense2SparseSRNNLayer.apply


class Dense2SparseSNNLayer(torch.autograd.Function):
    '''
    d2s-snn
    '''
    @staticmethod
    def forward(ctx, inputs, weight, nb_hidden, den2spa_output, prs):
        # weight_r, spk_rec, ctx=context
        batch_size = inputs.shape[0]
        nb_inputs = inputs.shape[2]  # Batch, Time, Units
        nb_steps = inputs.shape[1]

        beta = float(np.exp(-float(prs['time_step']) / float(prs['tau_mem'])))
        th = float(prs['th'])
        b_th = float(prs['b_th'])
        device = prs['device']
        reset_m = prs['reset_m']

        ain_idxs = torch.nonzero(inputs < 100.0)
        ain_i = ain_idxs[:, 2]

        ######### using to fix t_window
        # temp_beta_0dot9= float(np.exp(-0.5)) # 0.6065
        # t_window = torch.div(torch.log10(torch.tensor(prs.alpha_window)), \
        #   torch.log10(torch.tensor(temp_beta_0dot9)),rounding_mode='trunc')
        # print('beta: ', beta)
        # print('t_window: ', int(t_window))
        ######### using to fix t_window        END

        t_window = torch.div(torch.log10(torch.tensor(float(prs['alpha_window']))),
                             torch.log10(torch.tensor(beta)),
                             rounding_mode='trunc')

        # Forward
        with torch.no_grad():
            # + weight_bias
            # # Batch,Time,Input x Input,Output -> Batch,Time,Output
            h1 = torch.einsum('abc,cd->abd',
                              (inputs, weight))
            mem = torch.zeros((batch_size, den2spa_output),
                              device=device, dtype=torch.float)

            spk = torch.zeros((batch_size, den2spa_output),
                              device=device, dtype=torch.float)
            out = torch.zeros((batch_size, nb_hidden),
                              device=device, dtype=torch.float)

            trace = torch.zeros((batch_size, nb_inputs),
                                device=device, dtype=torch.float)

            mem_rec = [mem]
            spk_rec = [spk]
            spk_trace = [trace]

            # Membrane - th
            if reset_m == 'soft':
                for t in range(nb_steps - 1):
                    mem = beta * mem + h1[:, t, :] - out

                    mthr = mem - th
                    out = torch.zeros_like(mem)
                    out[mthr > 0] = 1.0
                    # ========== NEW STUFF ============= #
                    trace = beta * trace + inputs[:, t, :]
                    spk_trace.append(trace)
                    # =========================== #
                    mem_rec.append(mem)
                    spk_rec.append(out)

                mem_rec = torch.stack(mem_rec, dim=1)
                spk_rec = torch.stack(spk_rec, dim=1)
                spk_trace = torch.stack(spk_trace, dim=1)
                # ========= NEW STUFF =================== #
                # Indices of mem_rec above threshold mask [b, t, j]
                aout_idxs = torch.nonzero(torch.logical_and(mem_rec > b_th,\
                                        mem_rec < 2. * th - b_th))
                aout_mem = (mem_rec[aout_idxs[:, 0],
                            aout_idxs[:, 1], aout_idxs[:, 2]] - th)

            # Membrane set to 0
            elif reset_m == 'hard':
                last_mthr = torch.zeros_like(mem)
                for t in range(nb_steps - 1):
                    mem[last_mthr > 0] = 0.0  # hard reset
                    mem = beta * mem + h1[:, t, :]

                    mthr = mem - th
                    last_mthr = mthr  # record mthr
                    out = torch.zeros_like(mem)
                    out[mthr > 0] = 1.0
                    # ================= NEW STUFF ========== #
                    trace = beta * trace + inputs[:, t, :]
                    spk_trace.append(trace)
                    # ========================= #
                    mem_rec.append(mem)
                    spk_rec.append(out)

                mem_rec = torch.stack(mem_rec, dim=1)
                spk_rec = torch.stack(spk_rec, dim=1)
                spk_trace = torch.stack(spk_trace, dim=1)
                # ========================== NEW STUFF =============== #
                # Indices of mem_rec above threshold mask [b, t, j]
                aout_idxs = torch.nonzero(torch.logical_and(mem_rec > b_th,\
                                mem_rec < 2. * th - b_th))
                aout_mem = (mem_rec[aout_idxs[:, 0],
                            aout_idxs[:, 1], aout_idxs[:, 2]] - th)
            else:
                raise Exception('The reset mode is only supported \
                                by hard and soft, \
                                please check it in config.py')
            # Indices out
            aout_b = aout_idxs[:, 0]
            aout_t = aout_idxs[:, 1]
            aout_i = aout_idxs[:, 2]
            # Indices in
            # ============== #

        # Save for backward
        alphas = (torch.full((nb_steps,), beta, device=device) **
                  torch.arange(nb_steps, device=device))

        ctx.save_for_backward(spk_trace, aout_b, aout_t, aout_i, ain_i,
                              aout_mem, t_window, alphas, weight)
        # ctx.save_for_backward(spk_trace, aout_b, aout_t, aout_i, aout_mem, \
        #                       ain_b, ain_t, ain_i, alphas, weight, weight_r)

        ctx.batch_size = batch_size
        ctx.nb_steps = nb_steps
        ctx.nb_hidden = nb_hidden
        ctx.nb_inputs = nb_inputs
        ctx.den2spa_output = den2spa_output
        spk_rec.requires_grad = True
        aout_idxs.requires_grad = False
        return spk_rec, aout_idxs

    @staticmethod
    def backward(ctx, grad_output, grad_dummy1): 
        # ATTENTION! DO NOT delete grad_dummy1, \
        # this pramater is used to balance the grad number of input and output!
        spk_trace, aout_b, aout_t, aout_i, _, aout_mem, t_window, \
            alphas, weight = ctx.saved_tensors

        batch_size, nb_steps, nb_inputs, nb_hidden, den2spa_output = \
            ctx.batch_size, ctx.nb_steps, ctx.nb_inputs, \
                ctx.nb_hidden, ctx.den2spa_output
        dense_grad_output, grad_weights = None, None

        # Active output values gradient
        grad_output = grad_output[aout_b, aout_t, aout_i]
        grad_output_idxs = torch.nonzero(grad_output, as_tuple=True)[0]
        grad_output = grad_output[grad_output_idxs]
        aout_b = aout_b[grad_output_idxs]
        aout_t = aout_t[grad_output_idxs]
        aout_i = aout_i[grad_output_idxs]
        aout_mem = aout_mem[grad_output_idxs]
        if grad_output.numel() == 0:
            grad_weights = torch.zeros_like(weight)

            dense_grad_output = torch.zeros((batch_size, nb_steps, nb_inputs),
                                            device=grad_output.device)
            return dense_grad_output, grad_weights, None, None, None
            # return grad_output, grad_weights, None, None, None

        ds_out = surr_grad_spike_cuda(aout_mem, grad_output)  # ds/dv

        torch.cuda.synchronize()
        #weight_start = time.time()

        if ctx.needs_input_grad[2]:
            print('If this has been printed in the screen, \
                it means SRNNs Recurrent remain to deal!!! \
                This measage is created by Wang Minghao.')
            # grad_weights_r = s3gd_wR_backward_cuda(aout_b, aout_t,
            #                                        aout_i, ds_out,
            #                                        nb_hidden, nb_hidden)
        torch.cuda.synchronize()
        #weight_end_time = time.time()
        #weight_time = weight_end_time - weight_start
        # Weight gradient
        if ctx.needs_input_grad[1]:
            grad_weights = s3gd_backward_cuda(spk_trace, aout_b, aout_t,
                                              aout_i, ds_out,
                                              nb_inputs, nb_hidden)
        torch.cuda.synchronize()
        #wr_weight_end_time = time.time()
        #wr_time = wr_weight_end_time - weight_end_time
        # Input spikes gradient
        if ctx.needs_input_grad[0]:
            # This is false for first layer

            # Reorder active output indices to (b,j,t) format
            bjt = aout_b * nb_hidden * nb_steps + aout_i * nb_steps + aout_t
            bjt, bjt_idx = torch.sort(bjt)
            ds_out = ds_out[bjt_idx]

            # Find indices to start from and frequencies (output)
            b = torch.div(bjt, (nb_hidden * nb_steps), rounding_mode='trunc')
            j = torch.div((bjt - b * nb_hidden * nb_steps),
                          nb_steps, rounding_mode='trunc')
            ts_out = bjt - b * nb_hidden * nb_steps - j * nb_steps
            bj = b * nb_hidden + j
            # How many t for a given b and j
            aout_bj_freqs = torch.bincount(
                bj, minlength=batch_size * nb_hidden)
            idxs = torch.nonzero(aout_bj_freqs)  # Where to put the results
            ends = torch.cumsum(aout_bj_freqs[idxs], dim=0) - 1  # Indices
            bj_ends = torch.full((batch_size * nb_hidden,), -1, device='cuda')
            bj_ends[idxs] = ends

            #########################
            # total threads
            total_num_threads = bj.numel()
            ###############################################################

            # Call kernel to compute ds gradient
            t_window = int(t_window)
            dense_grad_output = s3gd_s_backward_master(
                # Computing
                ds_out,  # Values to compute deltas (\red{dS[bjt]})
                ts_out,
                bj_ends,
                bj,
                alphas,  # Powers of alpha
                # Constants
                total_num_threads,
                batch_size,
                nb_steps,
                den2spa_output,
                nb_hidden,
                t_window
            )

            dense_grad_output = torch.einsum('abd,cd->abc',
                                             dense_grad_output, weight)
        torch.cuda.synchronize()
        #s_end_time = time.time()
        #s_time = s_end_time - wr_weight_end_time
        # try:
        #     len(ain_i)
        # except TypeError:
        #     print(f'ain_i is {type(ain_i)}, \
        #           weight grad time is {weight_time}, \
        #           r_grad is {wr_time}, s_grad is {s_time}')
        # else:
        #     print(f'ain_i is {len(ain_i)}, \
        #           weight grad time is {weight_time}, \
        #           r_grad is {wr_time}, s_grad is {s_time}')

        return dense_grad_output, grad_weights, None, None, None


Dense2SparseSNNLayer = Dense2SparseSNNLayer.apply
