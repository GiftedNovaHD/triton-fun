import triton
import triton.language as tl

@triton.jit
def _ssnorm_residual_bwd(dx_ptr,
                         dresidual_input_ptr,
                         dgamma_partial_ptr,
                         dy_ptr,
                         dresidual_output_gradient_ptr,
                         residual_output_ptr,
                         inv_norm_ptr,
                         g_ptr,
                         row_stride,
                         feature_dim,
                         eps: tl.constexpr,
                         scale,
                         HAS_RESIDUAL_IN: tl.constexpr,
                         HAS_DRESIDUAL_OUT: tl.constexpr,
                         GROUP: tl.constexpr,
                         BLOCK: tl.constexpr,
                        ):
  """
  Fused backward pass for SSNorm with residual. 
  """
  row = tl.program_id(0)
  base = row * row_stride

  cols = tl.arange(0, BLOCK)
  mask = cols < feature_dim

  residual = tl.load(residual_output_ptr + base + cols,
                     mask = mask, 
                     other = 0.0,
                     eviction_policy = "evict_first",
                     # cache_modifier = ".ca",
                     # layout = tl.Layout.STRIDED,
                     ).to(tl.float32)
  dy = tl.load(dy_ptr + base + cols, 
               mask = mask,
               other = 0.0,
               eviction_policy = "evict_first",
               # cache_modifier = ".ca",
               # layout = tl.Layout.STRIDED,
               ).to(tl.float32)

  inv_norm = tl.load(inv_norm_ptr + row).to(tl.float32)
  denom = 1.0 / inv_norm

  # gamma = sqrt(D) * (g + 1)
  g = tl.load(g_ptr).to(tl.float32)
  gamma = (g + 1.0) * scale

  dot = tl.sum(dy * residual, axis = 0)

  # Match F.normalize clamp:
  is_clamped = denom <= tl.full([], eps, tl.float32)

  inv2 = inv_norm * inv_norm
  dresidual_from_y = tl.where(
    is_clamped,
    dy * gamma * inv_norm,
    gamma * inv_norm * (dy - residual * (inv2 * dot)),
  )

  # Add upstream gradient from prenorm residual output if it exist
  if HAS_DRESIDUAL_OUT:
    dresidual_out = tl.load(dresidual_output_gradient_ptr + base + cols,
                            mask = mask,
                            other = 0.0,
                            eviction_policy = "evict_first",
                            # cache_modifier = ".ca",
                            # layout = tl.Layout.STRIDED,
                            ).to(tl.float32)
    dresidual = dresidual_out + dresidual_from_y
  else:
    dresidual = dresidual_from_y

  # residual_out = x + residual_in => dx = dresidual, dresidual_in = dresidual
  tl.store(dx_ptr + base + cols, dresidual, mask = mask)
  if HAS_RESIDUAL_IN:
    tl.store(dresidual_input_ptr + base + cols, dresidual, mask = mask)

  # dg = d/dg [gamma] * sum(dy * residual * inv_norm)
  # gamma = scale * (g + 1) => dgamma/dg = scale
  dg_row = scale * (inv_norm * dot)

  bucket = row % GROUP
  tl.atomic_add(dgamma_partial_ptr + bucket, dg_row)
