#include <torch/extension.h>
#include <vector>


// CUDA declarations
torch::Tensor surr_grad_spike_cuda(torch::Tensor memth_out, torch::Tensor grad_out);
torch::Tensor s3gd_w_backward_cuda(torch::Tensor spk_trace,
                                 torch::Tensor aout_b,
                                 torch::Tensor aout_t,
                                 torch::Tensor aout_i,
                                 torch::Tensor ds_out,
                                 int nb_inputs,
                                 int nb_hidden);

torch::Tensor s3gd_wR_backward_cuda(torch::Tensor spk_trace_R,
                                  torch::Tensor aout_b,
                                  torch::Tensor aout_t,
                                  torch::Tensor aout_i,
                                  torch::Tensor ds_out,
                                  int nb_inputs,
                                  int nb_hidden);
////////////////dense2sparse
torch::Tensor s3gd_w_backward_Dense2Sparse_cuda(torch::Tensor spk_trace,
                                 torch::Tensor aout_b,
                                 torch::Tensor aout_t,
                                 torch::Tensor aout_i,
                                 torch::Tensor ds_out,
                                 int nb_inputs,
                                 int nb_hidden);

torch::Tensor s3gd_wR_backward_Dense2Sparse_cuda(torch::Tensor spk_trace_R,
                                  torch::Tensor aout_b,
                                  torch::Tensor aout_t,
                                  torch::Tensor aout_i,
                                  torch::Tensor ds_out,
                                  int nb_inputs,
                                  int nb_hidden);

torch::Tensor s3gd_w_self_R_backward_Dense2Sparse_cuda(torch::Tensor spk_trace_R,
                                  torch::Tensor aout_b,
                                  torch::Tensor aout_t,
                                  torch::Tensor aout_i,
                                  torch::Tensor ds_out,
                                  int nb_inputs,
                                  int nb_hidden);

////////////////////end/////
torch::Tensor s3gd_s_backward_master_cuda(
                                   torch::Tensor ds_out,
                                   torch::Tensor ts_out,
                                   torch::Tensor bj_ends,
                                   torch::Tensor aout_bj_freqs,
                                   torch::Tensor ts_in,
                                   torch::Tensor b_ends,
                                   torch::Tensor ain_b_freqs,
                                   torch::Tensor alphas,
                                   torch::Tensor weight,
                                   torch::Tensor b_in_unique_freqs,
                                   torch::Tensor bj_out_unique_freqs,
                                   torch::Tensor bj_out_unique,
                                   torch::Tensor ain_bt_freqs,
                                   torch::Tensor ain_bt_starts,
                                   torch::Tensor ain_i,
                                   const int total_num_threads,  // len(bj_out_unique)
                                   const int batch_size,
                                   const int nb_steps,
                                   const int nb_inputs,
                                   const int nb_hidden);

torch::Tensor s3gd_s_backward_Dense2Sparse_cuda(
                                   torch::Tensor ds_out,
                                   torch::Tensor ts_out,
                                   torch::Tensor bj_ends,
                                   torch::Tensor bj,
                                   torch::Tensor alphas,

                                   const int total_num_threads,  // len(bj_out_unique)
                                   const int batch_size,
                                   const int nb_steps,
                                   const int den2spa_output,
                                   const int nb_hidden,
                                   const int T_Window);
// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_1D(x) AT_ASSERTM(x.dim()==1, #x " must be 1D")
#define CHECK_LONG(x) AT_ASSERTM(x.dtype()==torch::kInt64, #x " must be LongTensor")

torch::Tensor surr_grad_spike(torch::Tensor memth_out, torch::Tensor grad_out) {
  CHECK_INPUT(memth_out);
  CHECK_INPUT(grad_out);
  return surr_grad_spike_cuda(memth_out, grad_out);
}

torch::Tensor s3gd_w_backward(
        torch::Tensor spk_trace,
        torch::Tensor aout_b,
        torch::Tensor aout_t,
        torch::Tensor aout_i,
        torch::Tensor ds_out,
        int nb_inputs,
        int nb_hidden
        ) {
  CHECK_INPUT(spk_trace);
  CHECK_INPUT(aout_b);
  CHECK_INPUT(aout_t);
  CHECK_INPUT(aout_i);
  CHECK_INPUT(ds_out);
  return s3gd_w_backward_cuda(spk_trace, aout_b, aout_t, aout_i, ds_out,
                            nb_inputs, nb_hidden);
}
torch::Tensor s3gd_wR_backward(
        torch::Tensor spk_trace_R,
        torch::Tensor aout_b,
        torch::Tensor aout_t,
        torch::Tensor aout_i,
        torch::Tensor ds_out,
        int nb_inputs,
        int nb_hidden
        ) {
  CHECK_INPUT(spk_trace_R);
  CHECK_INPUT(aout_b);
  CHECK_INPUT(aout_t);
  CHECK_INPUT(aout_i);
  CHECK_INPUT(ds_out);
  return s3gd_wR_backward_cuda(spk_trace_R, aout_b, aout_t, aout_i, ds_out,
                            nb_inputs, nb_hidden);
}

/////////////////////DEnse2sparse
torch::Tensor s3gd_w_backward_Dense2Sparse(
        torch::Tensor spk_trace,
        torch::Tensor aout_b,
        torch::Tensor aout_t,
        torch::Tensor aout_i,
        torch::Tensor ds_out,
        int nb_inputs,
        int nb_hidden
        ) {
  CHECK_INPUT(spk_trace);
  CHECK_INPUT(aout_b);
  CHECK_INPUT(aout_t);
  CHECK_INPUT(aout_i);
  CHECK_INPUT(ds_out);
  return s3gd_w_backward_Dense2Sparse_cuda(spk_trace, aout_b, aout_t, aout_i, ds_out,
                            nb_inputs, nb_hidden);
}
torch::Tensor s3gd_wR_backward_Dense2Sparse(
        torch::Tensor spk_trace_R,
        torch::Tensor aout_b,
        torch::Tensor aout_t,
        torch::Tensor aout_i,
        torch::Tensor ds_out,
        int nb_inputs,
        int nb_hidden
        ) {
  CHECK_INPUT(spk_trace_R);
  CHECK_INPUT(aout_b);
  CHECK_INPUT(aout_t);
  CHECK_INPUT(aout_i);
  CHECK_INPUT(ds_out);
  return s3gd_wR_backward_Dense2Sparse_cuda(spk_trace_R, aout_b, aout_t, aout_i, ds_out,
                            nb_inputs, nb_hidden);
}
torch::Tensor s3gd_w_self_R_backward_Dense2Sparse(
        torch::Tensor spk_trace_R,
        torch::Tensor aout_b,
        torch::Tensor aout_t,
        torch::Tensor aout_i,
        torch::Tensor ds_out,
        int nb_inputs,
        int nb_hidden
        ) {
  CHECK_INPUT(spk_trace_R);
  CHECK_INPUT(aout_b);
  CHECK_INPUT(aout_t);
  CHECK_INPUT(aout_i);
  CHECK_INPUT(ds_out);
  return s3gd_w_self_R_backward_Dense2Sparse_cuda(spk_trace_R, aout_b, aout_t, aout_i, ds_out,
                            nb_inputs, nb_hidden);
}
///////////////end/////////

torch::Tensor s3gd_s_backward_master(
                                    // Computing deltas
                                    torch::Tensor ds_out,  // Values to compute deltas (\red{dS[bjt]})
                                    torch::Tensor ts_out,  // Times active output values to compute deltas
                                    torch::Tensor bj_ends,  // Index fot ts_out with last time to read. Points to last time for each bj
                                    torch::Tensor aout_bj_freqs,  // How many t we need to compute for a given b&j
                                    // Recording deltas
                                    torch::Tensor ts_in,  // Times to record
                                    torch::Tensor b_ends,  // Index for ts_in with last time to record each batch
                                    torch::Tensor ain_b_freqs,  // How many t we need to record for a given batch
                                    // Other
                                    torch::Tensor alphas,  // Precomputed powers of alpha
                                    torch::Tensor weight,
                                    // Writing deltas
                                    torch::Tensor b_in_unique_freqs, // b_in_unique_freqs[b] tells how many steps will each of the j elements in this b will take
                                    torch::Tensor bj_out_unique_freqs, // bj_out_unique_freqs[bj] tells how many bj we had before this given bj in the current batch
                                    torch::Tensor bj_out_unique,
                                    // Computing gradient
                                   torch::Tensor ain_bt_freqs,
                                   torch::Tensor ain_bt_starts,
                                   torch::Tensor ain_i,
                                    // Constants
                                    const int total_num_threads,  // len(bj_out_unique)
                                    const int batch_size,
                                    const int nb_steps,
                                    const int nb_inputs,
                                    const int nb_hidden) {
        CHECK_INPUT(ds_out);
        CHECK_INPUT(ts_out);
        CHECK_INPUT(bj_ends);
        CHECK_INPUT(aout_bj_freqs);
        CHECK_INPUT(ts_in);
        CHECK_INPUT(b_ends);
        CHECK_INPUT(ain_b_freqs);
        CHECK_INPUT(alphas);
        CHECK_INPUT(weight);
        CHECK_INPUT(b_in_unique_freqs);
        CHECK_INPUT(bj_out_unique_freqs);
        CHECK_INPUT(bj_out_unique);
        CHECK_INPUT(ain_bt_freqs);
        CHECK_INPUT(ain_bt_starts);
        CHECK_INPUT(ain_i);
        // std::cout << "running s3gd_s_backward_master" << std::endl;
        // std::cout << "bj_out_unique_freqs:" << bj_out_unique_freqs << std::endl;

        return s3gd_s_backward_master_cuda(ds_out, ts_out, bj_ends, aout_bj_freqs,
                                          ts_in, b_ends, ain_b_freqs,
                                          alphas, weight, b_in_unique_freqs, bj_out_unique_freqs, bj_out_unique,
                                          ain_bt_freqs, ain_bt_starts, ain_i,
                                          total_num_threads, batch_size, nb_steps, nb_inputs, nb_hidden);
    }

torch::Tensor s3gd_s_backward_Dense2Sparse(
                                    // Computing deltas
                                    torch::Tensor ds_out,  // Values to compute deltas (\red{dS[bjt]})
                                    torch::Tensor ts_out,  // Times active output values to compute deltas
                                    torch::Tensor bj_ends,  // Index fot ts_out with last time to read. Points to last time for each bj
                                    torch::Tensor bj,  // How many t we need to compute for a given b&j
                                    // Recording deltas
                                    // Other
                                    torch::Tensor alphas,  // Precomputed powers of alpha
                                    // Constants
                                    const int total_num_threads,  // len(bj_out_unique)
                                    const int batch_size,
                                    const int nb_steps,
                                    const int den2spa_output,
                                    const int nb_hidden,
                                    const int T_Window) {
        CHECK_INPUT(ds_out);
        CHECK_INPUT(ts_out);
        CHECK_INPUT(bj_ends);
        CHECK_INPUT(bj);
        CHECK_INPUT(alphas);
        // std::cout << "running s3gd_s_backward_Dense2Sparse" << std::endl;
        return s3gd_s_backward_Dense2Sparse_cuda(ds_out, ts_out, bj_ends, bj,
                                          alphas,
                                          total_num_threads, batch_size, nb_steps, den2spa_output, nb_hidden, T_Window);
    }


// Binder
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("surr_grad_spike", &surr_grad_spike, "surr_grad_spike");
  m.def("s3gd_w_backward", &s3gd_w_backward, "s3gd_w_backward");
  m.def("s3gd_wR_backward", &s3gd_wR_backward, "s3gd_wR_backward");
  m.def("s3gd_s_backward_master", &s3gd_s_backward_master, "s3gd_s_backward_master");

  m.def("s3gd_w_backward_Dense2Sparse", &s3gd_w_backward_Dense2Sparse, "s3gd_w_backward_Dense2Sparse");
  m.def("s3gd_wR_backward_Dense2Sparse", &s3gd_wR_backward_Dense2Sparse, "s3gd_wR_backward_Dense2Sparse");
  m.def("s3gd_w_self_R_backward_Dense2Sparse", &s3gd_w_self_R_backward_Dense2Sparse, "s3gd_w_self_R_backward_Dense2Sparse");
  m.def("s3gd_s_backward_Dense2Sparse", &s3gd_s_backward_Dense2Sparse, "s3gd_s_backward_Dense2Sparse");

}
