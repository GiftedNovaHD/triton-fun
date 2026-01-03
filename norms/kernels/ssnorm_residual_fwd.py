import triton
import triton.language as tl

@triton.jit
def _ssnorm_residual_fwd(x_ptr,
                         residual_input_ptr, # May be dummy variable if HAS_RESIDUAL_IN is False
                         y_ptr,
                         residual_output_ptr, 
                         inv_norm_ptr,
                         gamma_ptr,
                         row_stride,
                         feature_dim,
                         eps,
                         scale,
                         HAS_RESIDUAL_IN: tl.constexpr,
                         BLOCK: tl.constexpr,
                         ):
  """
  Fused forward pass for SSNorm with residual. Conceptually, the reference implementation computes
  the residual, norm, denominator, output, (and optionally residual output) in multiple kernels. 

  This version loads everything into registers/SRAM and fuses the residual addition, such that we
  don't ever materialize an intermediate `x + residual` tensor via a separate kernel. Essentially,
  a single kernel launch performs the residual, normalization, and scale all in SRAM/register. 
  """
  row = tl.program_id(0)
  base = row * row_stride

  cols = tl.arange(0, BLOCK)
  mask = cols < feature_dim

  # Load input (+ an optional residual) in FP32
  x = tl.load(x_ptr + base + cols,
              mask = mask, other = 0.0,
              eviction_policy = "evict_first",
              # cache_modifier = ".ca",
              # layout = tl.Layout.STRIDED,
              ).to(tl.float32)

  if HAS_RESIDUAL_IN:
    r = tl.load(residual_input_ptr + base + cols,
                mask = mask, other = 0.0,
                eviction_policy = "evict_first",
                # cache_modifier = ".ca",
                # layout = tl.Layout.STRIDED,
                ).to(tl.float32)
    residual = x + r
  else:
    residual = x
  
  # Save residual (in caller-chosen dtype: bf16/fp16 or fp32)
  tl.store(residual_output_ptr + base + cols, residual, mask = mask)

  # L2 norm + clamp
  sum_of_squares = tl.sum(residual * residual, axis = 0)
  norm = tl.sqrt(sum_of_squares)
  denom = tl.maximum(norm, tl.full([], eps, tl.float32))
  inv_norm = 1.0 / denom

  tl.store(inv_norm_ptr + row, inv_norm)
  
  # Apply gamma
  g = tl.load(gamma_ptr).to(tl.float32)
  gamma = (g + 1.0) * scale

  y = residual * inv_norm * gamma
  tl.store(y_ptr + base + cols, y, mask = mask)