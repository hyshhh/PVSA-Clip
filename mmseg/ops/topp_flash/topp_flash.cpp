#include <torch/extension.h>

// 旧版接口（兼容性）
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
                                      int64_t width);

// 优化版 v2：使用 shared memory
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
                                         int64_t width);

// 优化版 v3：使用寄存器 + warp shuffle（推荐）
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
                                         int64_t width);

torch::Tensor topp_flash_forward(torch::Tensor q_pix,
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
  TORCH_CHECK(q_pix.is_cuda(), "q_pix must be a CUDA tensor");
  TORCH_CHECK(kv_pix.is_cuda(), "kv_pix must be a CUDA tensor");
  TORCH_CHECK(r_weight.is_cuda(), "r_weight must be a CUDA tensor");
  TORCH_CHECK(r_idx.is_cuda(), "r_idx must be a CUDA tensor");
  TORCH_CHECK(keep_len.is_cuda(), "keep_len must be a CUDA tensor");
  TORCH_CHECK(r_idx.scalar_type() == torch::kLong, "r_idx must be int64");
  TORCH_CHECK(keep_len.scalar_type() == torch::kLong, "keep_len must be int64");
  TORCH_CHECK(keep_len.dim() == 2, "keep_len must be a 2D tensor");
  TORCH_CHECK(keep_len.size(0) == q_pix.size(0) &&
                  keep_len.size(1) == q_pix.size(1),
              "keep_len shape must match q_pix n and p2");

  // 检查 dtype 支持
  auto dtype = q_pix.scalar_type();
  TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
              "q_pix must be float32, float16, or bfloat16");
  TORCH_CHECK(kv_pix.scalar_type() == dtype, "kv_pix dtype must match q_pix");
  TORCH_CHECK(r_weight.scalar_type() == dtype, "r_weight dtype must match q_pix");

  return topp_flash_forward_cuda_v3(q_pix.contiguous(),
                                    kv_pix.contiguous(),
                                    r_weight.contiguous(),
                                    r_idx.contiguous(),
                                    keep_len.contiguous(),
                                    num_heads,
                                    qk_dim,
                                    dim,
                                    scale,
                                    n_win,
                                    height,
                                    width);
}

// 保留旧版接口
torch::Tensor topp_flash_forward_legacy(torch::Tensor q_pix,
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
  TORCH_CHECK(q_pix.is_cuda(), "q_pix must be a CUDA tensor");
  TORCH_CHECK(kv_pix.scalar_type() == torch::kFloat32,
              "legacy forward only supports float32");

  return topp_flash_forward_cuda(q_pix.contiguous(),
                                 kv_pix.contiguous(),
                                 r_weight.contiguous(),
                                 r_idx.contiguous(),
                                 keep_len.contiguous(),
                                 num_heads, qk_dim, dim, scale, n_win, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topp_flash_forward, "PVSA topp flash forward (optimized, fp16/bf16/fp32)");
  m.def("forward_legacy", &topp_flash_forward_legacy, "PVSA topp flash forward (legacy, fp32 only)");
  m.def("forward_v2", &topp_flash_forward_cuda_v2, "PVSA topp flash forward v2 (shared memory)");
  m.def("forward_v3", &topp_flash_forward_cuda_v3, "PVSA topp flash forward v3 (register based)");
}
