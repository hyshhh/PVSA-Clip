#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

// ============================================================================
// 配置常量
// ============================================================================
constexpr int WARP_SIZE = 32;
constexpr int TILE_KV = 16;  // K/V tile 大小，可调

// ============================================================================
// 辅助函数：在线 softmax 更新
// ============================================================================
__device__ __forceinline__ void online_softmax_update(
    float &m_prev, float &l_prev, float &o_prev,
    float s_new, float v_new) {
  float m_new = fmaxf(m_prev, s_new);
  float exp_prev = expf(m_prev - m_new);
  float exp_new = expf(s_new - m_new);
  l_prev = l_prev * exp_prev + exp_new;
  o_prev = o_prev * exp_prev + exp_new * v_new;
  m_prev = m_new;
}

// ============================================================================
// 核心 kernel：一个 warp 算一个 (coarse, head, q_pos)，输出多个 V 通道
// 使用在线 softmax，K/V tiling
// 限制：head_q <= 64, head_v <= 256
// ============================================================================
template <typename scalar_t>
__global__ void topp_flash_forward_kernel_v2(
    const scalar_t *__restrict__ q_pix,
    const scalar_t *__restrict__ kv_pix,
    const scalar_t *__restrict__ r_weight,
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
    int64_t coarse_total) {
  // Grid: (coarse_total, num_heads, q_len)
  // Block: (WARP_SIZE,) - 一个 warp
  const int64_t coarse = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t q_pos = blockIdx.z;

  if (coarse >= coarse_total) return;

  const int64_t batch = coarse / p2;
  const int64_t p = coarse % p2;
  const int64_t head_v = dim / num_heads;
  const int64_t head_q = qk_dim / num_heads;
  const int64_t route_base = coarse * topk;
  const int tid = threadIdx.x;

  // 检查限制
  if (head_q > 64 || head_v > 256) return;

  int64_t valid_topk = keep_len[coarse];
  if (valid_topk > topk) valid_topk = topk;
  if (valid_topk <= 0) {
    for (int64_t v_ch = tid; v_ch < head_v; v_ch += WARP_SIZE) {
      int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim +
                        head * head_v + v_ch;
      out[out_idx] = 0.0f;
    }
    return;
  }

  // Q 向量：使用寄存器存储（head_q <= 64）
  const int64_t q_base = ((batch * p2 + p) * q_len + q_pos) * qk_dim + head * head_q;
  float q_local[64];
  for (int64_t d = 0; d < head_q; d++) {
    q_local[d] = static_cast<float>(q_pix[q_base + d]);
  }

  // 在线 softmax 状态
  float m_prev = -CUDART_INF_F;
  float l_prev = 0.0f;
  
  // 每个线程负责的 V 通道数
  const int v_per_thread = (head_v + WARP_SIZE - 1) / WARP_SIZE;
  float o_prev[8];  // head_v <= 256, v_per_thread <= 8
  for (int i = 0; i < v_per_thread; i++) o_prev[i] = 0.0f;

  // K/V tile 循环
  const int64_t total_kv = valid_topk * kv_len;
  
  for (int64_t tile_start = 0; tile_start < total_kv; tile_start += TILE_KV) {
    const int64_t tile_end = min(tile_start + TILE_KV, total_kv);
    const int tile_size = static_cast<int>(tile_end - tile_start);

    float s_tile[TILE_KV];
    float v_tile[TILE_KV][8];
    
    for (int t = 0; t < tile_size; t++) {
      const int64_t global_kv = tile_start + t;
      const int64_t tk = global_kv / kv_len;
      const int64_t kv_pos = global_kv % kv_len;

      const int64_t kv_window = r_idx[route_base + tk];
      const float route_weight = static_cast<float>(r_weight[route_base + tk]);
      const int64_t kv_base = ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);

      float score = 0.0f;
      for (int64_t d = 0; d < head_q; d++) {
        float k_val = static_cast<float>(kv_pix[kv_base + head * head_q + d]) * route_weight;
        score += q_local[d] * k_val;
      }
      s_tile[t] = score * scale;

      for (int64_t v_ch = tid; v_ch < head_v; v_ch += WARP_SIZE) {
        float v_val = static_cast<float>(kv_pix[kv_base + qk_dim + head * head_v + v_ch]) * route_weight;
        v_tile[t][v_ch / WARP_SIZE] = v_val;
      }
    }

    for (int t = 0; t < tile_size; t++) {
      float m_new = fmaxf(m_prev, s_tile[t]);
      float exp_prev = expf(m_prev - m_new);
      float exp_new = expf(s_tile[t] - m_new);
      l_prev = l_prev * exp_prev + exp_new;
      for (int i = 0; i < v_per_thread; i++) {
        o_prev[i] = o_prev[i] * exp_prev + exp_new * v_tile[t][i];
      }
      m_prev = m_new;
    }
  }

  for (int64_t v_ch = tid; v_ch < head_v; v_ch += WARP_SIZE) {
    int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim +
                      head * head_v + v_ch;
    int i = v_ch / WARP_SIZE;
    out[out_idx] = o_prev[i] / fmaxf(l_prev, 1.0e-20f);
  }
}

// ============================================================================
// 优化的 kernel：使用 warp shuffle 减少共享内存访问
// 限制：head_q <= 64, head_v <= 32
// ============================================================================
template <typename scalar_t>
__global__ void topp_flash_forward_kernel_v3(
    const scalar_t *__restrict__ q_pix,
    const scalar_t *__restrict__ kv_pix,
    const scalar_t *__restrict__ r_weight,
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
    int64_t coarse_total) {
  // Grid: (coarse_total * num_heads * q_len,)
  // Block: (WARP_SIZE,)
  const int64_t idx = blockIdx.x;
  const int64_t coarse = idx / (num_heads * q_len);
  const int64_t head_qpos = idx % (num_heads * q_len);
  const int64_t head = head_qpos / q_len;
  const int64_t q_pos = head_qpos % q_len;

  if (coarse >= coarse_total) return;

  const int64_t batch = coarse / p2;
  const int64_t p = coarse % p2;
  const int64_t head_v = dim / num_heads;
  const int64_t head_q = qk_dim / num_heads;
  const int64_t route_base = coarse * topk;
  const int tid = threadIdx.x;

  // 检查限制
  if (head_q > 64 || head_v > 32) return;

  int64_t valid_topk = keep_len[coarse];
  if (valid_topk > topk) valid_topk = topk;
  if (valid_topk <= 0) {
    if (tid < head_v) {
      int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim +
                        head * head_v + tid;
      out[out_idx] = 0.0f;
    }
    return;
  }

  // Q 向量：寄存器存储
  const int64_t q_base = ((batch * p2 + p) * q_len + q_pos) * qk_dim + head * head_q;
  float q_local[64];
  for (int64_t d = 0; d < head_q; d++) {
    q_local[d] = static_cast<float>(q_pix[q_base + d]);
  }

  // 在线 softmax 状态
  float m_prev = -CUDART_INF_F;
  float l_prev = 0.0f;
  float o_prev = 0.0f;

  // 每个线程负责一个 V 通道
  const int64_t my_v_ch = tid;

  for (int64_t tk = 0; tk < valid_topk; tk++) {
    const int64_t kv_window = r_idx[route_base + tk];
    const float route_weight = static_cast<float>(r_weight[route_base + tk]);

    for (int64_t kv_pos = 0; kv_pos < kv_len; kv_pos++) {
      const int64_t kv_base = ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);

      float score = 0.0f;
      for (int64_t d = 0; d < head_q; d++) {
        float k_val = static_cast<float>(kv_pix[kv_base + head * head_q + d]) * route_weight;
        score += q_local[d] * k_val;
      }
      score *= scale;

      float v_val = 0.0f;
      if (my_v_ch < head_v) {
        v_val = static_cast<float>(kv_pix[kv_base + qk_dim + head * head_v + my_v_ch]) * route_weight;
      }

      float m_new = fmaxf(m_prev, score);
      float exp_prev = expf(m_prev - m_new);
      float exp_new = expf(score - m_new);
      l_prev = l_prev * exp_prev + exp_new;
      o_prev = o_prev * exp_prev + exp_new * v_val;
      m_prev = m_new;
    }
  }

  if (my_v_ch < head_v) {
    int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim +
                      head * head_v + my_v_ch;
    out[out_idx] = o_prev / fmaxf(l_prev, 1.0e-20f);
  }
}

// ============================================================================
// 旧的 kernel（保留兼容性）
// ============================================================================
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
                                          int64_t coarse_total) {
  const int64_t coarse = blockIdx.x;
  if (coarse >= coarse_total) return;

  const int64_t batch = coarse / p2;
  const int64_t p = coarse % p2;
  const int64_t head_v = dim / num_heads;
  const int64_t head_q = qk_dim / num_heads;
  const int64_t route_base = coarse * topk;
  const int64_t outputs_per_window = q_len * dim;

  int64_t valid_topk = keep_len[coarse];
  if (valid_topk > topk) valid_topk = topk;

  for (int64_t local = threadIdx.x; local < outputs_per_window; local += blockDim.x) {
    const int64_t c_out = local % dim;
    const int64_t q_pos = local / dim;
    const int64_t head = c_out / head_v;
    const int64_t head_c_out = c_out % head_v;
    const int64_t linear = coarse * outputs_per_window + local;

    if (valid_topk <= 0) {
      out[linear] = 0.0f;
      continue;
    }

    const int64_t q_base = (coarse * q_len + q_pos) * qk_dim;
    float max_score = -CUDART_INF_F;

    for (int64_t tk = 0; tk < valid_topk; ++tk) {
      const int64_t kv_window = r_idx[route_base + tk];
      const float route_weight = r_weight[route_base + tk];

      for (int64_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
        const int64_t k_base = ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);
        float score = 0.0f;

        for (int64_t d = 0; d < head_q; ++d) {
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

    for (int64_t tk = 0; tk < valid_topk; ++tk) {
      const int64_t kv_window = r_idx[route_base + tk];
      const float route_weight = r_weight[route_base + tk];

      for (int64_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
        const int64_t kv_base = ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);
        float score = 0.0f;

        for (int64_t d = 0; d < head_q; ++d) {
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

// ============================================================================
// unflatten kernel（不变）
// ============================================================================
__global__ void unflatten_windows_kernel(const float *__restrict__ flat,
                                         float *__restrict__ out,
                                         int64_t n,
                                         int64_t n_win,
                                         int64_t height,
                                         int64_t width,
                                         int64_t dim,
                                         int64_t total) {
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) return;

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

  int64_t flat_index = ((batch * n_win * n_win + p) * (q_h * q_w) + q_pos) * dim + c;
  out[linear] = flat[flat_index];
}

// ============================================================================
// 入口函数：优化版（v3 - 推荐）
// ============================================================================
torch::Tensor topp_flash_forward_cuda_v3(torch::Tensor q_pix,
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
                                         int64_t width) {
  const auto n = q_pix.size(0);
  const auto p2 = q_pix.size(1);
  const auto q_len = q_pix.size(2);
  const auto kv_len = kv_pix.size(2);
  const auto topk = r_idx.size(2);

  auto flat_out = torch::empty({n, p2, q_len, dim}, q_pix.options().dtype(torch::kFloat32));
  auto out = torch::empty({n, height, width, dim}, q_pix.options().dtype(torch::kFloat32));

  const int64_t coarse_total = n * p2;
  const int64_t total_blocks = coarse_total * num_heads * q_len;

  // 每个 block 一个 warp
  const int threads = WARP_SIZE;

  if (q_pix.scalar_type() == torch::kFloat16) {
    topp_flash_forward_kernel_v3<__half><<<static_cast<int>(total_blocks), threads, 0,
                                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q_pix.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kv_pix.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(r_weight.data_ptr<at::Half>()),
        r_idx.data_ptr<int64_t>(),
        keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), n_win, height, width, coarse_total);
  } else if (q_pix.scalar_type() == torch::kBFloat16) {
    topp_flash_forward_kernel_v3<__nv_bfloat16><<<static_cast<int>(total_blocks), threads, 0,
                                                   at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_pix.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kv_pix.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(r_weight.data_ptr<at::BFloat16>()),
        r_idx.data_ptr<int64_t>(),
        keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), n_win, height, width, coarse_total);
  } else {
    // float32 路径
    topp_flash_forward_kernel_v3<float><<<static_cast<int>(total_blocks), threads, 0,
                                          at::cuda::getCurrentCUDAStream()>>>(
        q_pix.data_ptr<float>(),
        kv_pix.data_ptr<float>(),
        r_weight.data_ptr<float>(),
        r_idx.data_ptr<int64_t>(),
        keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), n_win, height, width, coarse_total);
  }

  const int64_t out_total = n * height * width * dim;
  const int out_blocks = static_cast<int>((out_total + 255) / 256);
  unflatten_windows_kernel<<<out_blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      flat_out.data_ptr<float>(),
      out.data_ptr<float>(),
      n, n_win, height, width, dim, out_total);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ============================================================================
// 入口函数：使用寄存器的版本（v2 - 适合 head_q <= 64, head_v <= 256）
// ============================================================================
torch::Tensor topp_flash_forward_cuda_v2(torch::Tensor q_pix,
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
                                         int64_t width) {
  const auto n = q_pix.size(0);
  const auto p2 = q_pix.size(1);
  const auto q_len = q_pix.size(2);
  const auto kv_len = kv_pix.size(2);
  const auto topk = r_idx.size(2);

  auto flat_out = torch::empty({n, p2, q_len, dim}, q_pix.options().dtype(torch::kFloat32));
  auto out = torch::empty({n, height, width, dim}, q_pix.options().dtype(torch::kFloat32));

  const int64_t coarse_total = n * p2;
  const dim3 grid(coarse_total, num_heads, q_len);
  const int threads = WARP_SIZE;

  if (q_pix.scalar_type() == torch::kFloat16) {
    topp_flash_forward_kernel_v2<__half><<<grid, threads, 0,
                                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q_pix.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kv_pix.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(r_weight.data_ptr<at::Half>()),
        r_idx.data_ptr<int64_t>(),
        keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), n_win, height, width, coarse_total);
  } else if (q_pix.scalar_type() == torch::kBFloat16) {
    topp_flash_forward_kernel_v2<__nv_bfloat16><<<grid, threads, 0,
                                                   at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_pix.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kv_pix.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(r_weight.data_ptr<at::BFloat16>()),
        r_idx.data_ptr<int64_t>(),
        keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), n_win, height, width, coarse_total);
  } else {
    topp_flash_forward_kernel_v2<float><<<grid, threads, 0,
                                          at::cuda::getCurrentCUDAStream()>>>(
        q_pix.data_ptr<float>(),
        kv_pix.data_ptr<float>(),
        r_weight.data_ptr<float>(),
        r_idx.data_ptr<int64_t>(),
        keep_len.data_ptr<int64_t>(),
        flat_out.data_ptr<float>(),
        n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
        static_cast<float>(scale), n_win, height, width, coarse_total);
  }

  const int64_t out_total = n * height * width * dim;
  const int out_blocks = static_cast<int>((out_total + 255) / 256);
  unflatten_windows_kernel<<<out_blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      flat_out.data_ptr<float>(),
      out.data_ptr<float>(),
      n, n_win, height, width, dim, out_total);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ============================================================================
// 入口函数：旧版（兼容性）
// ============================================================================
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
                                      int64_t width) {
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
      n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
      static_cast<float>(scale), n_win, height, width, coarse_total);

  const int64_t out_total = n * height * width * dim;
  const int out_blocks = static_cast<int>((out_total + threads - 1) / threads);
  unflatten_windows_kernel<<<out_blocks, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      flat_out.data_ptr<float>(),
      out.data_ptr<float>(),
      n, n_win, height, width, dim, out_total);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
