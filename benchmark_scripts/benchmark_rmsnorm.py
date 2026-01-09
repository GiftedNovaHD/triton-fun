import torch
import triton
import triton.testing
from torch import Tensor

try:
  from mamba_ssm.ops.triton.layer_norm import rms_norm_fn
  HAS_MAMBA = True
except ImportError:
  HAS_MAMBA = False
  print("Mamba implementation not found. Skipping Mamba benchmark.")

from ops.rmsnorm import RMSNormTriton

# -----------------------
# Reference RMSNorm (Torch)
# -----------------------
def rmsnorm_ref(x: Tensor, weight: Tensor, bias=None, residual=None, prenorm=True, residual_in_fp32=False, eps=1e-6):
  assert x.ndim >= 1
  D = x.shape[-1]
  assert weight.shape == (D,)
  if bias is not None:
      assert bias.shape == (D,)

  # Form residual stream u
  if residual is None:
      u = x
  else:
      u = x + residual  # compute in x.dtype (matches typical fused kernel behavior)

  # res_out for prenorm path (optionally FP32)
  if prenorm:
      res_out = u.float() if residual_in_fp32 else u
  else:
      res_out = None

  # Compute rstd in FP32 for stability
  u_fp32 = u.float()
  mean_square = (u_fp32 * u_fp32).mean(dim=-1, keepdim=True)
  rstd = torch.rsqrt(mean_square + eps)  # FP32

  # Normalize and apply affine
  y = (u_fp32 * rstd).to(x.dtype) * weight
  if bias is not None:
      y = y + bias

  return (y, res_out) if prenorm else y

def rmsnorm_triton(x, weight, bias=None, residual=None, prenorm=True, residual_in_fp32=False, eps=1e-6):
  return RMSNormTriton.apply(x, weight, bias, residual, prenorm, residual_in_fp32, eps)

compiled_rmsnorm_ref = torch.compile(rmsnorm_ref)

# -----------------------
# Correctness test
# -----------------------
def test_rmsnorm(M, N, dtype, eps=1e-6, device="cuda"):
  print(f"Testing RMSNorm correctness: M={M}, N={N}, dtype={dtype}")
  torch.manual_seed(0)

  # Inputs
  x = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)
  weight = torch.randn((N,), dtype=dtype, device=device).requires_grad_(True)
  bias = torch.randn((N,), dtype=dtype, device=device).requires_grad_(True)
  residual = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)

  # Output grads
  dy = torch.randn_like(x)
  dres_out = torch.randn_like(x)

  # --- Reference ---
  x_ref = x.detach().clone().requires_grad_(True)
  w_ref = weight.detach().clone().requires_grad_(True)
  b_ref = bias.detach().clone().requires_grad_(True)
  r_ref = residual.detach().clone().requires_grad_(True)

  y_ref, res_ref = rmsnorm_ref(x_ref, w_ref, b_ref, r_ref, prenorm=True, eps=eps)
  torch.autograd.backward([y_ref, res_ref], [dy, dres_out], retain_graph=False)

  dx_ref = x_ref.grad
  dw_ref = w_ref.grad
  db_ref = b_ref.grad
  dr_ref = r_ref.grad

  # --- Triton ---
  x_tri = x.detach().clone().requires_grad_(True)
  w_tri = weight.detach().clone().requires_grad_(True)
  b_tri = bias.detach().clone().requires_grad_(True)
  r_tri = residual.detach().clone().requires_grad_(True)

  y_tri, res_tri = rmsnorm_triton(x_tri, w_tri, b_tri, r_tri, prenorm=True, eps=eps)
  torch.autograd.backward([y_tri, res_tri], [dy, dres_out], retain_graph=False)

  dx_tri = x_tri.grad
  dw_tri = w_tri.grad
  db_tri = b_tri.grad
  dr_tri = r_tri.grad

  # --- Mamba (if available) ---
  if HAS_MAMBA:
      x_mamba = x.detach().clone().requires_grad_(True)
      w_mamba = weight.detach().clone().requires_grad_(True)
      b_mamba = bias.detach().clone().requires_grad_(True)
      r_mamba = residual.detach().clone().requires_grad_(True)

      # Mamba's rms_norm_fn signature might vary, usually:
      # rms_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6)
      y_mamba, res_mamba = rms_norm_fn(x_mamba, w_mamba, b_mamba, residual=r_mamba, prenorm=True, eps=eps)
      torch.autograd.backward([y_mamba, res_mamba], [dy, dres_out], retain_graph=False)

      # Check vs Mamba
      if dtype in (torch.float16, torch.bfloat16):
          atol, rtol = 1e-2, 1e-2
      else:
          atol, rtol = 1e-4, 1e-4
          
      try:
          assert torch.allclose(y_tri, y_mamba, atol=atol, rtol=rtol), "y mismatch vs Mamba"
          assert torch.allclose(res_tri, res_mamba, atol=atol, rtol=rtol), "res_out mismatch vs Mamba"
          # Note: Gradients might differ slightly due to implementation details (e.g. reduction order)
          # but usually should be close.
          print("Matches Mamba implementation.")
      except AssertionError as e:
          print(f"Mamba mismatch: {e}")

  # Check vs Reference
  if dtype in (torch.float16, torch.bfloat16):
      atol, rtol = 1e-2, 1e-2
  else:
      atol, rtol = 1e-4, 1e-4

  assert torch.allclose(y_tri, y_ref, atol=atol, rtol=rtol), f"y mismatch (max diff: {(y_tri - y_ref).abs().max()})"
  assert torch.allclose(res_tri, res_ref, atol=atol, rtol=rtol), f"res_out mismatch (max diff: {(res_tri - res_ref).abs().max()})"
  assert torch.allclose(dx_tri, dx_ref, atol=atol, rtol=rtol), f"dx mismatch (max diff: {(dx_tri - dx_ref).abs().max()})"
  assert torch.allclose(dw_tri, dw_ref, atol=atol, rtol=rtol), f"dw mismatch (max diff: {(dw_tri - dw_ref).abs().max()})"
  assert torch.allclose(db_tri, db_ref, atol=atol, rtol=rtol), f"db mismatch (max diff: {(db_tri - db_ref).abs().max()})"
  assert torch.allclose(dr_tri, dr_ref, atol=atol, rtol=rtol), f"dr mismatch (max diff: {(dr_tri - dr_ref).abs().max()})"

  print("PASSED ✅")


# -----------------------
# Benchmark
# -----------------------
line_vals = ["triton", "torch", "torch_compile"]
line_names = ["Triton", "Torch", "Torch Compile"]
if HAS_MAMBA:
  line_vals.append("mamba")
  line_names.append("Mamba")

@triton.testing.perf_report(
  triton.testing.Benchmark(
      x_names=["N"],
      x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
      line_arg="provider",
      line_vals=line_vals,
      line_names=line_names,
      styles=[("blue", "-"), ("green", "-"), ("red", "--"), ("orange", "-")],
      ylabel="GB/s (proxy)",
      plot_name="rmsnorm-fwd_bwd-step",
      args={"M": 4096, "dtype": torch.float16, "mode": "train_step"},
  )
)
def bench_rmsnorm(M, N, dtype, provider, mode="train_step", eps=1e-6, device="cuda"):
  x = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)
  residual = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)
  weight = torch.ones((N,), dtype=dtype, device=device).requires_grad_(True)
  bias = torch.zeros((N,), dtype=dtype, device=device).requires_grad_(True)

  dy = 0.1 * torch.randn_like(x)
  dres_out = 0.1 * torch.randn_like(x)

  quantiles = [0.5, 0.2, 0.8]

  def fwd():
      if provider == "triton":
          return rmsnorm_triton(x, weight, bias, residual, prenorm=True, eps=eps)
      if provider == "torch":
          return rmsnorm_ref(x, weight, bias, residual, prenorm=True, eps=eps)
      if provider == "torch_compile":
          return compiled_rmsnorm_ref(x, weight, bias, residual, prenorm=True, eps=eps)
      if provider == "mamba":
          return rms_norm_fn(x, weight, bias, residual=residual, prenorm=True, eps=eps)
      raise ValueError(provider)

  # warmups
  for _ in range(3):
      y, r = fwd()
      if mode == "train_step":
          torch.autograd.backward([y, r], [dy, dres_out], retain_graph=False)
      for t in [x, residual, weight, bias]:
          if t.grad is not None:
              t.grad = None

  if mode == "forward":
      # proxy bytes: read x+res (2) + read w+b (negligible) + write y+res_out (2) -> 4x tensor size
      def gbps(ms):
          return 4 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
      ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=100)
      return gbps(ms), gbps(max_ms), gbps(min_ms)

  # mode == "train_step"
  def train_step():
      y, r = fwd()
      torch.autograd.backward([y, r], [dy, dres_out], retain_graph=False)

  # proxy bytes for full step:
  # Forward: read x, res (2), write y, res_out (2)
  # Backward: read dy, dres_out (2), read x, res, rstd (recompute or saved), write dx, dr (2), write dw, db (negligible)
  # Total approx 8x tensor size
  def gbps(ms):
      return 8 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

  grads_to_none = [x, residual, weight, bias]

  ms, min_ms, max_ms = triton.testing.do_bench(
      train_step,
      quantiles=quantiles,
      grad_to_none=grads_to_none,
      rep=100,
  )
  return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
  # Correctness checks
  print("--- Correctness Tests ---")
  test_rmsnorm(32, 1024, torch.float32)
  test_rmsnorm(32, 1024, torch.float16)
  test_rmsnorm(256, 256, torch.float32)

  # Test larger N to trigger multi-pass kernel logic
  test_rmsnorm(32, 4096, torch.float32)
  test_rmsnorm(32, 8192, torch.float32)
  test_rmsnorm(4096, 8192, torch.float32)
  test_rmsnorm(4096, 16384, torch.float32)
  # Benchmarks
  print("\n--- Running Benchmarks ---")
  bench_rmsnorm.run(save_path=".", print_data=True)