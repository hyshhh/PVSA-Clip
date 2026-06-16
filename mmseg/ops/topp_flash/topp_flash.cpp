#include <torch/extension.h>
#include <pybind11/stl.h>

#include <vector>

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

torch::Tensor topp_fused_route_flash_forward_cuda(torch::Tensor route_query,
                                                  torch::Tensor q_pix,
                                                  torch::Tensor kv_pix,
                                                  int64_t topk,
                                                  double route_p,
                                                  double route_temperature,
                                                  double route_energy,
                                                  double route_scale,
                                                  double attn_scale,
                                                  int64_t num_heads,
                                                  int64_t qk_dim,
                                                  int64_t dim,
                                                  int64_t n_win,
                                                  int64_t height,
                                                  int64_t width);

std::vector<torch::Tensor> topp_route_forward_cuda(torch::Tensor query,
                                                   int64_t topk,
                                                   double p,
                                                   double temperature,
                                                   double energy,
                                                   double scale);

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
  return topp_flash_forward_cuda(
      q_pix.contiguous(), kv_pix.contiguous(), r_weight.contiguous(),
      r_idx.contiguous(), keep_len.contiguous(), num_heads, qk_dim, dim,
      scale, n_win, height, width);
}

torch::Tensor topp_fused_route_flash_forward(torch::Tensor route_query,
                                             torch::Tensor q_pix,
                                             torch::Tensor kv_pix,
                                             int64_t topk,
                                             double route_p,
                                             double route_temperature,
                                             double route_energy,
                                             double route_scale,
                                             double attn_scale,
                                             int64_t num_heads,
                                             int64_t qk_dim,
                                             int64_t dim,
                                             int64_t n_win,
                                             int64_t height,
                                             int64_t width) {
  TORCH_CHECK(route_query.is_cuda(), "route_query must be a CUDA tensor");
  TORCH_CHECK(q_pix.is_cuda(), "q_pix must be a CUDA tensor");
  TORCH_CHECK(kv_pix.is_cuda(), "kv_pix must be a CUDA tensor");
  return topp_fused_route_flash_forward_cuda(
      route_query.contiguous(), q_pix.contiguous(), kv_pix.contiguous(),
      topk, route_p, route_temperature, route_energy, route_scale, attn_scale,
      num_heads, qk_dim, dim, n_win, height, width);
}

std::vector<torch::Tensor> topp_route_forward(torch::Tensor query,
                                              int64_t topk,
                                              double p,
                                              double temperature,
                                              double energy,
                                              double scale) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  return topp_route_forward_cuda(query.contiguous(), topk, p, temperature,
                                 energy, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topp_flash_forward,
        "PVSA topp flash attention forward (inference)");
  m.def("route_forward", &topp_route_forward,
        "PVSA topp route forward (inference)");
  m.def("fused_forward", &topp_fused_route_flash_forward,
        "PVSA fused topp route and flash forward (inference)");
}
