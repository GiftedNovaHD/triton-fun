"""
Fused implementation of Single-Scaled RMS Normalization.

Optimized implementation of Single-Scaled RMS Normalization with forward and backward passes. 

SSNorm adds a scaling parameter gamma across dimensions to remedy instability caused by the
scaling factor sqrt(d) across dimensions as in RMSNorm.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from torch import Tensor
from typing import Optional

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
              ).to(tl.float32)

  if HAS_RESIDUAL_IN:
    r = tl.load(residual_input_ptr + base + cols,
                mask = mask, other = 0.0,
                eviction_policy = "evict_first",
                # cache_modifier = ".ca",
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
                     ).to(tl.float32)
  dy = tl.load(dy_ptr + base + cols, 
               mask = mask,
               other = 0.0,
               eviction_policy = "evict_first",
               # cache_modifier = ".ca",
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

@triton.jit
def _reduce_dg_buckets(dg_partial_ptr,
                       dg_output_ptr,
                       num_buckets,
                       BLOCK: tl.constexpr,
                       ):
  """
  This is a reduction kernel that helps to sum `GROUP` number of partial scalars into a single scalar. 
  We do this to avoid the problem of scalar atomic contention, that is, every row computing a
  tl.atomic_add(dg_ptr, dg_row) which would mean tons of programs trying to access the same address, leading
  to serialized atomics, and slow down the entire kernel as a result. 

  1. Allocate GROUP buckets 
  2. Each row will atomic_add into bucket = row % GROUP 
  3. Computation is done in GROUP addresses. 
  """
  offset = tl.arange(0, BLOCK)
  mask = offset < num_buckets
  values = tl.load(dg_partial_ptr + offset, mask = mask, other = 0.0)
  total = tl.sum(values, axis = 0)
  tl.store(dg_output_ptr, total)

class SSNorm(nn.Module):
  def __init__(self,
               hidden_dim: int,
               eps: float = 1e-6
              ):
    """
    This is a Single-Scaled RMS Normalization module
    Compared to normal RMSNorm, instead of each channel having its own scaling factor,
    SSNorm shares a single scaling factor across all channels.
    This can be thought of as projecting a d-dimensional vector onto a d-1 dimensional hypersphere with radius g where g is the scaling factor.

    This implementation will use a fused Triton kernel when possible, which fuses the adding of the residual and the normalization.

    Args:
      - `hidden_dim` (`int`): Hidden dimension size
      - `eps` (`float`): Epsilon value for normalization
    """
    super().__init__()
    self.eps = eps
    self.scale = hidden_dim ** 0.5
    self.g = nn.Parameter(torch.zeros(1))
    nn.init.constant_(self.g, 0)

  def forward(self, 
              x: Tensor, 
              residual: Optional[Tensor] = None,
              prenorm: bool = True,
              residual_in_fp32: bool = False,
              eps: float = 1e-6
              ) -> Tensor:
    """
    Args:
      - `x` (`Tensor`): Input tensor of shape (`batch`, `sequence_length`, `hidden_dim`)
      - `residual` (`Optional[Tensor]`): Residual tensor of shape (`batch`, `sequence_length`, `hidden_dim`)
      - `prenorm` (`bool`): This normalization is used as pre-norm, if True, the residual will be returned.
      - `residual_in_fp32` (`bool`): Whether to perform residual in fp32
      - `eps` (`float`): Epsilon value for normalization

    Returns:
      - `hidden_states` (`Tensor`): Normalized hidden states of shape (`batch`, `sequence_length`, `hidden_dim`)
      - `residual` (`Tensor`): Residual tensor of shape (`batch`, `sequence_length`, `hidden_dim`) if `prenorm` is True, otherwise not returned.
    """
    if x.is_cuda:
      return SSNormTriton.apply(x, self.g, residual, prenorm, residual_in_fp32, self.eps)

    # Perform normalization in fp32 for stability
    if residual is None:
      residual = x.to(torch.float32) if residual_in_fp32 else x
    else:
      residual = residual + x
    
    hidden_states = F.normalize(residual.float(), p = 2, dim = -1, eps = self.eps).to(residual.dtype)

    if prenorm is True:
      return hidden_states * self.scale * (self.g + 1), residual
    else:
      return hidden_states * self.scale * (self.g + 1)

class SSNormTriton(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, g, residual = None, prenorm = True, residual_in_fp32 = False, eps = 1e-6):
    assert x.is_cuda
    assert x.shape[-1] > 0
    D = x.shape[-1]
    scale = float(D ** 0.5)

    # Fast-path assumption: last dim contiguous (otherwise reshape may copy!)
    if x.stride(-1) != 1:
      res = x if residual is None else (x + residual)
      res_fp = res.float()
      y = F.normalize(res_fp, p = 2, dim = -1, eps = eps).to(x.dtype) * scale * (g + 1)
      if prenorm:
        return y, res.float() if residual_in_fp32 else res
      return y
    
    x2d = x.reshape(-1, D)
    rows = x2d.shape[0] 

    if residual is None:
      res_in2d = x2d
      has_residual_in = False
    else:
      assert residual.shape == x.shape
      if residual.stride(-1) != 1:
        res = x if residual is None else (x + residual)
        res_fp = res.float()
        y = F.normalize(res_fp, p = 2, dim = -1, eps = eps).to(x.dtype) * scale * (g + 1)
        if prenorm:
          return y, res.float() if residual_in_fp32 else res
        return y
      res_in2d = residual.reshape(-1, D)
      has_residual_in = True
    
    # Store residual in fp32 if passed as option
    res_out_dtype = torch.float32 if residual_in_fp32 else x.dtype

    y2d = torch.empty_like(x2d, dtype = x.dtype)
    res_out2d = torch.empty_like(x2d, dtype = res_out_dtype)
    inv_norm = torch.empty((rows,), dtype = torch.float32, device = x.device)

    BLOCK = 1024 if D <= 1024 else triton.next_power_of_2(D)
    num_warps = 4 if BLOCK <= 1024 else 8

    _ssnorm_residual_fwd[(rows,)](
      x2d, res_in2d, y2d, res_out2d, inv_norm, g,
      x2d.stride(0),
      D,
      eps = eps,
      scale = scale,
      HAS_RESIDUAL_IN = has_residual_in,
      BLOCK = BLOCK,
      num_warps = num_warps,
    )

    ctx.save_for_backward(res_out2d, inv_norm, g)
    ctx.D = D
    ctx.scale = scale
    ctx.eps = eps
    ctx.has_residual_in = has_residual_in
    ctx.prenorm = prenorm
    ctx.BLOCK = BLOCK
    ctx.num_warps = num_warps
    ctx.x_dtype = x.dtype
    ctx.res_out_dtype = res_out_dtype

    y = y2d.reshape_as(x)
    res_out = res_out2d.reshape_as(x)
    return (y, res_out) if prenorm else y
  
  @staticmethod
  def backward(ctx, *grad_outputs):
    res_out2d, inv_norm, g = ctx.saved_tensors
    D = ctx.D
    scale = ctx.scale
    eps = ctx.eps
    rows = res_out2d.shape[0]

    dy = grad_outputs[0]
    dy2d = dy.reshape(-1, D)

    if ctx.prenorm and len(grad_outputs) > 1 and (grad_outputs[1] is not None):
      dres_out = grad_outputs[1].reshape(-1, D)
      has_dresidual_out = True
    else:
      dres_out = dy2d
      has_dresidual_out = False
    
    dx2d = torch.empty_like(dy2d, dtype = ctx.x_dtype)

    if ctx.has_residual_in:
      dres_in2d = torch.empty_like(dy2d, dtype = ctx.x_dtype)
    else:
      dres_in2d = dx2d

    if D <= 1024:
      group = 256
    elif D <= 4096:
      group = 128
    elif D <= 8192:
      group = 96
    else:
      group = 64
    
    group = min(group, rows) if rows > 0 else 1

    dg_partial = torch.zeros((group,), dtype = torch.float32, device = dy.device)
    dg_out = torch.empty_like(g, dtype = torch.float32)

    _ssnorm_residual_bwd[(rows,)](
      dx2d, dres_in2d, dg_partial,
      dy2d, dres_out, res_out2d, inv_norm, g,
      dy2d.stride(0),
      D,
      eps = eps,
      scale = scale,
      HAS_RESIDUAL_IN = ctx.has_residual_in,
      HAS_DRESIDUAL_OUT = has_dresidual_out,
      GROUP = group,
      BLOCK = ctx.BLOCK,
      num_warps = ctx.num_warps,
    )

    reduction_block = triton.next_power_of_2(group)
    _reduce_dg_buckets[(1,)](
      dg_partial, dg_out, group,
      BLOCK = reduction_block,
      num_warps = 1,
    )

    dx = dx2d.reshape(-1, D).reshape_as(grad_outputs[0])

    if ctx.has_residual_in:
      dresidual = dres_in2d.reshape(-1, D).reshape_as(grad_outputs[0])
    else:
      dresidual = None

    return dx, dg_out, dresidual, None, None, None