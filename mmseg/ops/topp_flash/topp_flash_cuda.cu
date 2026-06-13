#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

// ============================================================================
// 配置
// ============================================================================
constexpr int WARP_SIZE = 32;
constexpr int TILE_KV = 16;

// ============================================================================
// Warp reduce sum
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// ============================================================================
// Block reduce sum（模板化，支持不同 block 大小）
// ============================================================================
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  
  val = warp_reduce_sum(val);
  
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  
  val = (threadIdx.x < NUM_WARPS) ? shared[threadIdx.x] : 0.0f;
  if (wid == 0) val = warp_reduce_sum(val);
  
  return val;
}

// ============================================================================
// 高性能 kernel（模板化 block 大小）
// ============================================================================
template <typename scalar_t, int BLOCK_SIZE>
__global__ void topp_flash_kernel(
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
    int64_t coarse_total) {
  
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
  
  __shared__ float smem[BLOCK_SIZE];
  __shared__ float s_q[128];  // head_q <= 128
  
  // 协作加载 Q
  const int64_t q_base = ((batch * p2 + p) * q_len + q_pos) * qk_dim + head * head_q;
  for (int d = tid; d < head_q; d += BLOCK_SIZE) {
    s_q[d] = static_cast<float>(q_pix[q_base + d]);
  }
  __syncthreads();
  
  int64_t valid_topk = keep_len[coarse];
  if (valid_topk > topk) valid_topk = topk;
  
  const int v_per_thread = (head_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float m_prev = -__int_as_float(0x7f800000);
  float l_prev = 0.0f;
  
  if (valid_topk <= 0) {
    for (int i = 0; i < v_per_thread; i++) {
      int64_t v_ch = tid + i * BLOCK_SIZE;
      if (v_ch < head_v) {
        int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim + head * head_v + v_ch;
        out[out_idx] = 0.0f;
      }
    }
    return;
  }
  
  const int64_t total_kv = valid_topk * kv_len;
  
  for (int64_t tile_start = 0; tile_start < total_kv; tile_start += TILE_KV) {
    const int64_t tile_end = min(tile_start + TILE_KV, total_kv);
    const int tile_size = static_cast<int>(tile_end - tile_start);
    
    for (int t = 0; t < tile_size; t++) {
      const int64_t global_kv = tile_start + t;
      const int64_t tk = global_kv / kv_len;
      const int64_t kv_pos = global_kv % kv_len;
      
      const int64_t kv_window = r_idx[route_base + tk];
      const float route_weight = static_cast<float>(r_weight[route_base + tk]);
      const int64_t kv_base = ((batch * p2 + kv_window) * kv_len + kv_pos) * (qk_dim + dim);
      
      // 协作计算 Q*K
      float partial_score = 0.0f;
      for (int64_t d = tid; d < head_q; d += BLOCK_SIZE) {
        float k_val = static_cast<float>(kv_pix[kv_base + head * head_q + d]) * route_weight;
        partial_score += s_q[d] * k_val;
      }
      
      float score = block_reduce_sum<BLOCK_SIZE>(partial_score, smem);
      score *= scale;
      
      // 在线 softmax
      float m_new = fmaxf(m_prev, score);
      float exp_prev = expf(m_prev - m_new);
      float exp_new = expf(score - m_new);
      l_prev = l_prev * exp_prev + exp_new;
      
      for (int i = 0; i < v_per_thread; i++) {
        int64_t v_ch = tid + i * BLOCK_SIZE;
        float v_val = 0.0f;
        if (v_ch < head_v) {
          v_val = static_cast<float>(kv_pix[kv_base + qk_dim + head * head_v + v_ch]) * route_weight;
        }
        o_acc[i] = o_acc[i] * exp_prev + exp_new * v_val;
      }
      
      m_prev = m_new;
    }
  }
  
  for (int i = 0; i < v_per_thread; i++) {
    int64_t v_ch = tid + i * BLOCK_SIZE;
    if (v_ch < head_v) {
      int64_t out_idx = ((batch * p2 + p) * q_len + q_pos) * dim + head * head_v + v_ch;
      out[out_idx] = o_acc[i] / fmaxf(l_prev, 1.0e-20f);
    }
  }
}

// ============================================================================
// unflatten kernel
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
// 入口函数：根据 head_q 动态选择 block 大小
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
  const int64_t head_q = qk_dim / num_heads;

  auto flat_out = torch::empty({n, p2, q_len, dim}, q_pix.options().dtype(torch::kFloat32));
  auto out = torch::empty({n, height, width, dim}, q_pix.options().dtype(torch::kFloat32));

  const int64_t coarse_total = n * p2;
  const dim3 grid(coarse_total, num_heads, q_len);
  auto stream = at::cuda::getCurrentCUDAStream();

  // 根据 head_q 选择 block 大小
  // head_q <= 64  → 128 线程
  // head_q <= 128 → 256 线程
  const int threads = (head_q <= 64) ? 128 : 256;

  if (q_pix.scalar_type() == torch::kFloat16) {
    if (threads == 128) {
      topp_flash_kernel<__half, 128><<<grid, 128, 0, stream>>>(
          reinterpret_cast<const __half*>(q_pix.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(kv_pix.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(r_weight.data_ptr<at::Half>()),
          r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
          flat_out.data_ptr<float>(),
          n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
          static_cast<float>(scale), coarse_total);
    } else {
      topp_flash_kernel<__half, 256><<<grid, 256, 0, stream>>>(
          reinterpret_cast<const __half*>(q_pix.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(kv_pix.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(r_weight.data_ptr<at::Half>()),
          r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
          flat_out.data_ptr<float>(),
          n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
          static_cast<float>(scale), coarse_total);
    }
  } else if (q_pix.scalar_type() == torch::kBFloat16) {
    if (threads == 128) {
      topp_flash_kernel<__nv_bfloat16, 128><<<grid, 128, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q_pix.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(kv_pix.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(r_weight.data_ptr<at::BFloat16>()),
          r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
          flat_out.data_ptr<float>(),
          n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
          static_cast<float>(scale), coarse_total);
    } else {
      topp_flash_kernel<__nv_bfloat16, 256><<<grid, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q_pix.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(kv_pix.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(r_weight.data_ptr<at::BFloat16>()),
          r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
          flat_out.data_ptr<float>(),
          n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
          static_cast<float>(scale), coarse_total);
    }
  } else {
    if (threads == 128) {
      topp_flash_kernel<float, 128><<<grid, 128, 0, stream>>>(
          q_pix.data_ptr<float>(), kv_pix.data_ptr<float>(),
          r_weight.data_ptr<float>(),
          r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
          flat_out.data_ptr<float>(),
          n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
          static_cast<float>(scale), coarse_total);
    } else {
      topp_flash_kernel<float, 256><<<grid, 256, 0, stream>>>(
          q_pix.data_ptr<float>(), kv_pix.data_ptr<float>(),
          r_weight.data_ptr<float>(),
          r_idx.data_ptr<int64_t>(), keep_len.data_ptr<int64_t>(),
          flat_out.data_ptr<float>(),
          n, p2, q_len, kv_len, topk, num_heads, qk_dim, dim,
          static_cast<float>(scale), coarse_total);
    }
  }

  const int64_t out_total = n * height * width * dim;
  const int out_blocks = static_cast<int>((out_total + 255) / 256);
  unflatten_windows_kernel<<<out_blocks, 256, 0, stream>>>(
      flat_out.data_ptr<float>(), out.data_ptr<float>(),
      n, n_win, height, width, dim, out_total);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
