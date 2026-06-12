#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace
{

  __global__ void topp_flash_forward_kernel(const float *__restrict__ q_pix,
                                            const float *__restrict__ kv_pix,
                                            const float *__restrict__ r_weight,
                                            const int64_t *__restrict__ r_idx,
                                            const int64_t *__restrict__ keep_len,
                                            float *__restrict__ out,
                                            int64_t n,
                                            int64_t p2,
                                            int64_t q_len,
                                            int64_t kv_len,
                                            int64_t topk,
                                            int64_t num_heads,
                                            int64_t qk_dim,
                                            int64_t dim,
                                            float scale,
                                            int64_t n_win,
                                            int64_t height,
                                            int64_t width,
                                            int64_t coarse_total)
  {
    // One block owns one coarse routing query window: coarse = batch * p2 + p.
    const int64_t coarse = blockIdx.x;
    if (coarse >= coarse_total)
    {
      return;
    }

    const int64_t batch = coarse / p2;
    const int64_t p = coarse % p2;
    const int64_t head_v = dim / num_heads;
    const int64_t head_q = qk_dim / num_heads;
    const int64_t route_base = coarse * topk;
    const int64_t outputs_per_window = q_len * dim;

    int64_t valid_topk = keep_len[coarse];
    if (valid_topk > topk)
    {
      valid_topk = topk;
    }

    for (int64_t local = threadIdx.x; local < outputs_per_window;
         local += blockDim.x)
    {
      const int64_t c_out = local % dim;
      const int64_t q_pos = local / dim;
      const int64_t head = c_out / head_v;
      const int64_t head_c_out = c_out % head_v;
      const int64_t linear = coarse * outputs_per_window + local;

      if (valid_topk <= 0)
      {
        out[linear] = 0.0f;
        continue;
      }

      const int64_t q_base = (coarse * q_len + q_pos) * qk_dim;
      float max_score = -CUDART_INF_F;

      for (int64_t tk = 0; tk < valid_topk; ++tk)
      {
        const int64_t kv_window = r_idx[route_base + tk];
        const float route_weight = r_weight[route_base + tk];

        for (int64_t kv_pos = 0; kv_pos < kv_len; ++kv_pos)
        {
          const int64_t k_base =
              ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);
          float score = 0.0f;

          for (int64_t d = 0; d < head_q; ++d)
          {
            const float q_val = q_pix[q_base + head * head_q + d];
            const float k_val = kv_pix[k_base + head * head_q + d] * route_weight;
            score += q_val * k_val;
          }

          score *= scale;
          max_score = fmaxf(max_score, score);
        }
      }

      float denom = 0.0f;
      float value = 0.0f;

      for (int64_t tk = 0; tk < valid_topk; ++tk)
      {
        const int64_t kv_window = r_idx[route_base + tk];
        const float route_weight = r_weight[route_base + tk];

        for (int64_t kv_pos = 0; kv_pos < kv_len; ++kv_pos)
        {
          const int64_t kv_base =
              ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);
          float score = 0.0f;

          for (int64_t d = 0; d < head_q; ++d)
          {
            const float q_val = q_pix[q_base + head * head_q + d];
            const float k_val = kv_pix[kv_base + head * head_q + d] * route_weight;
            score += q_val * k_val;
          }

          score *= scale;
          const float prob_num = expf(score - max_score);
          denom += prob_num;

          const int64_t v_offset = qk_dim + head * head_v + head_c_out;
          const float v_val = kv_pix[kv_base + v_offset] * route_weight;
          value += prob_num * v_val;
        }
      }

      out[linear] = value / fmaxf(denom, 1.0e-20f);
    }
  }

  __global__ void unflatten_windows_kernel(const float *__restrict__ flat,
                                           float *__restrict__ out,
                                           int64_t n,
                                           int64_t n_win,
                                           int64_t height,
                                           int64_t width,
                                           int64_t dim,
                                           int64_t total)
  {
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total)
    {
      return;
    }

    int64_t c = linear % dim;
    int64_t w = (linear / dim) % width;
    int64_t h = (linear / (dim * width)) % height;
    int64_t batch = linear / (dim * width * height);

    int64_t q_h = height / n_win;
    int64_t q_w = width / n_win;
    int64_t win_y = h / q_h;
    int64_t win_x = w / q_w;
    int64_t local_y = h % q_h;
    int64_t local_x = w % q_w;
    int64_t p = win_y * n_win + win_x;
    int64_t q_pos = local_y * q_w + local_x;

    int64_t flat_index = ((batch * n_win * n_win + p) * (q_h * q_w) + q_pos) *
                             dim +
                         c;
    out[linear] = flat[flat_index];
  }

} // namespace

torch::Tensor topp_flash_forward_cuda(torch::Tensor q_pix,
                                      torch::Tensor kv_pix,
                                      torch::Tensor r_weight,
                                      torch::Tensor r_idx,
                                      torch::Tensor keep_len,
                                      int64_t num_heads,
                                      int64_t qk_dim,
                                      int64_t dim,
                                      double scale,
                                      int64_t n_win,
                                      int64_t height,
                                      int64_t width)
{
  const auto n = q_pix.size(0);
  const auto p2 = q_pix.size(1);
  const auto q_len = q_pix.size(2);
  const auto kv_len = kv_pix.size(2);
  const auto topk = r_idx.size(2);

  auto flat_out = torch::empty({n, p2, q_len, dim}, q_pix.options());
  auto out = torch::empty({n, height, width, dim}, q_pix.options());

  const int threads = 256;
  const int64_t coarse_total = n * p2;

  topp_flash_forward_kernel<<<static_cast<int>(coarse_total), threads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      q_pix.data_ptr<float>(),
      kv_pix.data_ptr<float>(),
      r_weight.data_ptr<float>(),
      r_idx.data_ptr<int64_t>(),
      keep_len.data_ptr<int64_t>(),
      flat_out.data_ptr<float>(),
      n,
      p2,
      q_len,
      kv_len,
      topk,
      num_heads,
      qk_dim,
      dim,
      static_cast<float>(scale),
      n_win,
      height,
      width,
      coarse_total);

  const int64_t out_total = n * height * width * dim;
  const int out_blocks = static_cast<int>((out_total + threads - 1) / threads);
  unflatten_windows_kernel<<<out_blocks, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      flat_out.data_ptr<float>(),
      out.data_ptr<float>(),
      n,
      n_win,
      height,
      width,
      dim,
      out_total);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
