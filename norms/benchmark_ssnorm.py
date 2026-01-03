import torch
import torch.nn.functional as F
import triton
import triton.testing

from mamba_ssm.ops.triton.layer_norm import rms_norm_fn
from norms.ssnorm import SSNormTriton

# Reference SSNorm (Torch)
def ssnorm_ref(x, g, residual=None, prenorm=True, residual_in_fp32=False, eps=1e-6):
  D = x.shape[-1]
  scale = D ** 0.5

  if residual is None:
      residual_out = x.to(torch.float32) if residual_in_fp32 else x
  else:
      residual_out = residual + x

  hs = F.normalize(residual_out.float(), p=2, dim=-1, eps=eps).to(residual_out.dtype)
  y = hs * scale * (g + 1)

  return (y, residual_out) if prenorm else y


def ssnorm_triton(x, g, residual=None, prenorm=True, residual_in_fp32=False, eps=1e-6):
  return SSNormTriton.apply(x, g, residual, prenorm, residual_in_fp32, eps)


compiled_ssnorm_ref = torch.compile(ssnorm_ref)

# Correctness test
def test_ssnorm(M, N, dtype, eps=1e-6, device="cuda"):
  print(f"Testing SSNorm correctness: M={M}, N={N}, dtype={dtype}")
  torch.manual_seed(0)

  x = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)
  residual = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)

  # keep g fp32 (recommended)
  g = torch.zeros((1,), dtype=torch.float32, device=device).requires_grad_(True)

  dy = torch.randn_like(x)
  dres_out = torch.randn_like(x)

  # --- reference ---
  x_ref = x.detach().clone().requires_grad_(True)
  r_ref = residual.detach().clone().requires_grad_(True)
  g_ref = g.detach().clone().requires_grad_(True)

  y_ref, res_ref = ssnorm_ref(x_ref, g_ref, r_ref, prenorm=True, eps=eps)
  torch.autograd.backward([y_ref, res_ref], [dy, dres_out], retain_graph=False)

  dx_ref = x_ref.grad
  dr_ref = r_ref.grad
  dg_ref = g_ref.grad

  # --- triton ---
  x_tri = x.detach().clone().requires_grad_(True)
  r_tri = residual.detach().clone().requires_grad_(True)
  g_tri = g.detach().clone().requires_grad_(True)

  y_tri, res_tri = ssnorm_triton(x_tri, g_tri, r_tri, prenorm=True, eps=eps)
  torch.autograd.backward([y_tri, res_tri], [dy, dres_out], retain_graph=False)

  dx_tri = x_tri.grad
  dr_tri = r_tri.grad
  dg_tri = g_tri.grad

  if dtype in (torch.float16, torch.bfloat16):
    atol, rtol = 1e-2, 1e-2
  else:
    atol, rtol = 1e-4, 1e-4

  assert torch.allclose(y_tri, y_ref.to(dtype), atol=atol, rtol=rtol), "y mismatch"
  assert torch.allclose(res_tri, res_ref.to(dtype), atol=atol, rtol=rtol), "res_out mismatch"
  assert torch.allclose(dx_tri, dx_ref.to(dtype), atol=atol, rtol=rtol), "dx mismatch"
  assert torch.allclose(dr_tri, dr_ref.to(dtype), atol=atol, rtol=rtol), "dresidual mismatch"
  assert torch.allclose(dg_tri, dg_ref, atol=atol, rtol=rtol), "dg mismatch"

  print("PASSED ✅")


# -----------------------
# Bench: forward-only + train-step (fwd+bwd)
# -----------------------
@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=["N"],
    x_vals=[1024 * i for i in range(1, 9)],
    line_arg="provider",
    line_vals=["triton", "torch", "mamba", "torch_compile"],
    line_names=["Triton", "Torch", "Mamba", "Torch Compile"],
    styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "--")],
    ylabel="GB/s (proxy)",
    plot_name="ssnorm-fwd_bwd-step",
    args={"M": 4096, "dtype": torch.float16, "mode": "train_step"},
  )
)
def bench_ssnorm(M, N, dtype, provider, mode="train_step", eps=1e-6, device="cuda"):
  x = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)
  residual = torch.randn((M, N), dtype=dtype, device=device).requires_grad_(True)

  # g fp32
  g = torch.zeros((1,), dtype=torch.float32, device=device).requires_grad_(True)

  # Mamba params (RMSNorm baseline)
  mamba_weight = torch.ones((N,), dtype=dtype, device=device).requires_grad_(True)
  mamba_bias = torch.zeros((N,), dtype=dtype, device=device).requires_grad_(True)

  dy = 0.1 * torch.randn_like(x)
  dres_out = 0.1 * torch.randn_like(x)

  quantiles = [0.5, 0.2, 0.8]

  def fwd():
    if provider == "triton":
      return ssnorm_triton(x, g, residual, prenorm=True, eps=eps)
    if provider == "torch":
      return ssnorm_ref(x, g, residual, prenorm=True, eps=eps)
    if provider == "torch_compile":
      return compiled_ssnorm_ref(x, g, residual, prenorm=True, eps=eps)
    if provider == "mamba":
      # RMSNorm baseline (not identical math to SSNorm)
      return rms_norm_fn(
        x,
        mamba_weight,
        mamba_bias,
        residual=residual,
        prenorm=True,
        eps=eps,
      )
    raise ValueError(provider)

  # warmups: compile/jit
  for _ in range(3):
    y, r = fwd()
    torch.autograd.backward([y, r], [dy, dres_out], retain_graph=False)
    for t in [x, residual, g, mamba_weight, mamba_bias]:
      if t.grad is not None:
        t.grad = None

  if mode == "forward":
    # proxy bytes: read x+res (2) write y+res_out (2) -> 4x tensor size
    gbps = lambda ms: 4 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=100)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

  # mode == "train_step": fresh fwd + bwd each rep (works with torch_compile)
  def train_step():
    y, r = fwd()
    torch.autograd.backward([y, r], [dy, dres_out], retain_graph=False)

  # proxy bytes for full step (rough):
  # read x,res (2) + write y,res_out (2) + read dy,dres_out (2) + write dx,dres (2) => 8x tensor size
  gbps = lambda ms: 8 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

  grads_to_none = [x, residual, g]
  if provider == "mamba":
      grads_to_none += [mamba_weight, mamba_bias]

  ms, min_ms, max_ms = triton.testing.do_bench(
    train_step,
    quantiles=quantiles,
    grad_to_none=grads_to_none,
    rep=100,
  )
  return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
  test_ssnorm(32, 1024, torch.float32)
  test_ssnorm(32, 1024, torch.float16)

  print("\nRunning benchmarks...")
  bench_ssnorm.run(save_path=".", print_data=True)