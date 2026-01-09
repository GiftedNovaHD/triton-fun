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

from torch import Tensor
from typing import Optional

from norms.kernels.ssnorm_residual_fwd import _ssnorm_residual_fwd
from norms.kernels.ssnorm_residual_bwd import _ssnorm_residual_bwd
from norms.kernels.ssnorm_reduce_buckets import _reduce_dg_buckets

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