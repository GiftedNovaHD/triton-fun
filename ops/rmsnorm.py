"""
This is a custom fused implementation of RMSNorm that beats Torch Compile and Mamba
up to a sequence-length of 8192. 

We refer to the Mamba implementation of RMSNorm as the baseline and include fused residual
addition for pre-normalization architectures.

skill issue so i loose at seq_lens of > 8192 
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
import triton
import triton.language as tl
from typing import Optional

class FusedRMSNorm(nn.Module):
  def __init__(self, hidden_dim: int, bias: bool = False, eps: float = 1e-6, device = None, dtype = None):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_dim, device = device, dtype = dtype))
    self.bias = nn.Parameter(torch.zeros(hidden_dim, device = device, dtype = dtype)) if bias else None
  
  def forward(self, x: Tensor, residual: Optional[Tensor] = None, prenorm: bool = True, residual_in_fp32: bool = False):
    return RMSNormTriton.apply(x, self.weight, self.bias, residual, prenorm, residual_in_fp32, self.eps)

MAX_FUSED_SIZE = 65536 # This is ideal based on testing on H200

@triton.jit
def _rmsnorm_fwd_kernel(x_ptr,
                        residual_input_ptr,
                        y_ptr,
                        residual_output_ptr,
                        rstd_ptr,
                        weight_ptr,
                        bias_ptr,
                        row_stride: tl.constexpr,
                        N,
                        eps,
                        HAS_RESIDUAL_IN: tl.constexpr,
                        HAS_BIAS: tl.constexpr,
                        BLOCK: tl.constexpr,
                        NUM_ITERS: tl.constexpr,
                        ONE_PASS: tl.constexpr,
                       ):
  """
  Fused RMSNorm Forward Kernel.
  """
  pid = tl.program_id(0)
  base = pid * row_stride

  # Yoloing it because if ONE_PASS is true, we fit everything in registers.
  # Works for H200! 
  if ONE_PASS:
    columns = tl.arange(0, BLOCK)
    mask = columns < N

    # Load input and optional residual
    x = tl.load(x_ptr + base + columns, 
                mask = mask,
                other = 0.0,
                eviction_policy = "evict_first",
                ).to(tl.float32)
    
    if HAS_RESIDUAL_IN:
      r = tl.load(residual_input_ptr + base + columns,
                  mask = mask,
                  other = 0.0,
                  eviction_policy = "evict_first",
                  ).to(tl.float32)
      u = x + r
    else:
      u = x
    
    # Compute RMS in FP32 for precision
    sum_squares = tl.sum(u * u, axis = 0)
    mean_squares = sum_squares / tl.full([], N, tl.float32)
    reciprocal_stddev = tl.math.rsqrt(mean_squares + eps)
    tl.store(rstd_ptr + pid, reciprocal_stddev)

    # Apply weight and optional bias
    w = tl.load(weight_ptr + columns,
                mask = mask,
                other = 0.0,
                cache_modifier = ".ca",
               ).to(tl.float32)
    y = u * reciprocal_stddev * w
    
    if HAS_BIAS:
      b = tl.load(bias_ptr + columns,
                  mask = mask,
                  other = 0.0,
                  cache_modifier = ".ca",
                  ).to(tl.float32)
      y = y + b
    
    # Store results
    tl.store(residual_output_ptr + base + columns, u, mask = mask)
    tl.store(y_ptr + base + columns, y, mask = mask)
    return
  
  # Basically when N > MAX_BLOCK, we split it into multiple passes to preserve performance
  # But I'm lazy to optimize it for larger sizes so suck thumb
  sum_squares_acc = tl.zeros([], dtype = tl.float32)
  for iter in tl.static_range(0, NUM_ITERS):
    offset = iter * BLOCK
    columns = offset + tl.arange(0, BLOCK)
    mask = columns < N

    x = tl.load(x_ptr + base + columns,
                mask = mask,
                other = 0.0,
                eviction_policy = "evict_first",
                ).to(tl.float32)
    if HAS_RESIDUAL_IN:
      r = tl.load(residual_input_ptr + base + columns,
                  mask = mask,
                  other = 0.0,
                  eviction_policy = "evict_first",
                  ).to(tl.float32)
      u = x + r
    else: 
      u = x
    
    sum_squares_acc += tl.sum(u * u, axis = 0)

  mean_squares = sum_squares_acc / N
  reciprocal_stddev = tl.math.rsqrt(mean_squares + eps)
  tl.store(rstd_ptr + pid, reciprocal_stddev)

  for iter in tl.static_range(0, NUM_ITERS):
    offset = iter * BLOCK
    columns = offset + tl.arange(0, BLOCK)
    mask = columns < N

    x = tl.load(x_ptr + base + columns,
                mask = mask,
                other = 0.0,
                eviction_policy = "evict_first",
                ).to(tl.float32)
    if HAS_RESIDUAL_IN:
      r = tl.load(residual_input_ptr + base + columns,
                  mask = mask,
                  other = 0.0,
                  eviction_policy = "evict_first",
                  ).to(tl.float32)
      u = x + r
    else:
      u = x
    
    w = tl.load(weight_ptr + columns,
                mask = mask,
                other = 0.0,
                cache_modifier = ".ca",
                ).to(tl.float32)
    y = u * reciprocal_stddev * w
    
    if HAS_BIAS:
      b = tl.load(bias_ptr + columns,
                  mask = mask,
                  other = 0.0,
                  cache_modifier = ".ca",
                  ).to(tl.float32)
      y = y + b
    
    tl.store(residual_output_ptr + base + columns, u, mask = mask)
    tl.store(y_ptr + base + columns, y, mask = mask)

@triton.jit
def _rmsnorm_bwd_c1_kernel(
    u_ptr, dy_ptr, weight_ptr, rstd_ptr, c1_ptr,
    stride_row, N,
    eps,
    BLOCK: tl.constexpr
):
    """
    Pass 1 of 2 for Large N Backward: Compute C1 (scalar term per row).
    Actually for RMSNorm: c1 = sum(xhat * wdy) / N
    where xhat = u * rstd, wdy = dy * w.
    """
    pid = tl.program_id(0)
    row_idx = pid 
    
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    
    u = tl.load(u_ptr + row_idx * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row_idx * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
    
    # dot = sum(u * rstd * dy * w)
    xhat = u * rstd
    wdy = dy * w
    dot = tl.sum(xhat * wdy, axis=0)
    
    c1 = dot / tl.full([], N, tl.float32)
    tl.store(c1_ptr + row_idx, c1)

@triton.jit
def _rmsnorm_bwd_final_kernel(u_ptr, dy_ptr, dresidual_output_ptr, dx_ptr, dresidual_input_ptr,
                              weight_ptr, rstd_ptr, c1_ptr, 
                              dweight_ptr, dbias_ptr,
                              stride_row, rows, N,
                              rows_per_program,
                              BLOCK: tl.constexpr,
                              HAS_BIAS: tl.constexpr,
                              HAS_DRESIDUAL_OUT: tl.constexpr,
                              HAS_RESIDUAL_IN: tl.constexpr,
                              ):
  """
  Pass 2 of 2 for Large N Backward: Compute dx, dw, db using precomputed C1.
  Tiled over both rows and columns.
  """
  pid_row = tl.program_id(0)
  pid_col = tl.program_id(1)
  
  row_start = pid_row * rows_per_program
  col_start = pid_col * BLOCK
  
  cols = col_start + tl.arange(0, BLOCK)
  mask = cols < N
  
  # Load weights for this tile
  w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
  
  dw_accum = tl.zeros((BLOCK,), dtype=tl.float32)
  if HAS_BIAS:
    db_accum = tl.zeros((BLOCK,), dtype=tl.float32)
      
  for i in range(0, rows_per_program):
    row_idx = row_start + i
    if row_idx < rows:
      offset = row_idx * stride_row + cols
      
      u = tl.load(u_ptr + offset, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
      dy = tl.load(dy_ptr + offset, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
      rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
      c1 = tl.load(c1_ptr + row_idx).to(tl.float32)
      
      xhat = u * rstd
      wdy = dy * w
      
      # dx = (wdy - xhat * c1) * rstd
      dx = (wdy - xhat * c1) * rstd
      
      if HAS_DRESIDUAL_OUT:
        dres = tl.load(dresidual_output_ptr + offset, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        dx += dres
      
      tl.store(dx_ptr + offset, dx, mask=mask)
      if HAS_RESIDUAL_IN:
        tl.store(dresidual_input_ptr + offset, dx, mask=mask)
      
      dw_accum += dy * xhat
      if HAS_BIAS:
        db_accum += dy

  # Atomic add to global weight grads
  tl.atomic_add(dweight_ptr + cols, dw_accum, mask=mask)
  if HAS_BIAS:
    tl.atomic_add(dbias_ptr + cols, db_accum, mask=mask)

@triton.jit
def _rmsnorm_bwd_loop_rows_kernel(u_ptr, dy_ptr, dresidual_output_ptr, dx_ptr, dresidual_input_ptr,
                                  weight_ptr, rstd_ptr, dweight_ptr, dbias_ptr,
                                  stride_row, rows, N,
                                  rows_per_program,
                                  BLOCK: tl.constexpr,
                                  HAS_BIAS: tl.constexpr,
                                  HAS_DRESIDUAL_OUT: tl.constexpr,
                                  HAS_RESIDUAL_IN: tl.constexpr,
                                  ):
  """
  Optimized Backward Kernel for Medium N (ONE_PASS).
  
  Instead of parallelizing rows within a block (which limits rows per block due to register pressure),
  this kernel loops over rows sequentially within the block.
  
  This allows us to process many rows per block (e.g. 50-100), drastically reducing the 
  size of the partial weight gradient buffer `_dw` and the cost of the final reduction.
  """
  pid = tl.program_id(0)
  row_start = pid * rows_per_program
  
  cols = tl.arange(0, BLOCK)
  mask = cols < N
  
  w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
  
  dw_accum = tl.zeros((BLOCK,), dtype=tl.float32)
  if HAS_BIAS:
    db_accum = tl.zeros((BLOCK,), dtype=tl.float32)
  
  invN = 1.0 / tl.full([], N, tl.float32)

  for i in range(0, rows_per_program):
    row_idx = row_start + i
    if row_idx < rows:
      offset = row_idx * stride_row + cols
      
      # Load row data
      u = tl.load(u_ptr + offset, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
      dy = tl.load(dy_ptr + offset, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
      rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
      
      # Compute dot = sum(u * rstd * dy * w)
      xhat = u * rstd
      wdy = dy * w
      dot = tl.sum(xhat * wdy, axis=0)
      c1 = dot * invN
      
      # Compute dx
      dx = (wdy - xhat * c1) * rstd
      
      if HAS_DRESIDUAL_OUT:
        dres = tl.load(dresidual_output_ptr + offset, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        dx += dres
          
      tl.store(dx_ptr + offset, dx, mask=mask)
      if HAS_RESIDUAL_IN:
        tl.store(dresidual_input_ptr + offset, dx, mask=mask)
          
      # Accumulate gradients
      dw_accum += dy * xhat
      if HAS_BIAS:
        db_accum += dy

  # Store accumulated gradients for this chunk of rows
  tl.store(dweight_ptr + pid * N + cols, dw_accum, mask=mask)
  if HAS_BIAS:
      tl.store(dbias_ptr + pid * N + cols, db_accum, mask=mask)

def _choose_block(N: int, dtype: torch.dtype) -> int:
  """
  Choose the block size for the kernel.
  We want to maximize the block size to perform the normalization in a single pass (ONE_PASS)
  if possible, as this avoids reloading data from global memory.
  
  Max block size limited by Shared Memory (typically 48KB-164KB).
  16384 elements * 4 bytes (float32) = 64KB. Shld be safe afaik. Definitey safe for big memory H200.
  """
  block = triton.next_power_of_2(N)
  return min(block, 16384)

def _choose_num_warps(block: int) -> int:
  if block >= 16384:
    return 16
  return min(max(block // 256, 1), 8)

class RMSNormTriton(torch.autograd.Function):
  @staticmethod
  def forward(ctx,
              x: Tensor,
              weight: Tensor,
              bias: Optional[Tensor] = None,
              residual: Optional[Tensor] = None,
              prenorm: bool = True,
              residual_in_fp32: bool = False,
              eps: float = 1e-6,
             ):
    assert x.is_cuda, "CUDA ONLY"
    assert x.shape[-1] > 0
    assert weight is not None and weight.is_cuda
    D = x.shape[-1]
    assert weight.numel() == D and weight.ndim == 1
    
    if weight.stride(-1) != 1:
      weight = weight.contiguous()

    has_bias = (bias is not None)
    if has_bias:
      assert bias.is_cuda and bias.ndim == 1 and bias.numel() == D
      if bias.stride(-1) != 1:
        bias = bias.contiguous()

    x2d = x.reshape(-1, D)
    if x2d.stride(-1) != 1:
      x2d = x2d.contiguous()
    rows = x2d.shape[0]

    has_residual_in = (residual is not None)
    if has_residual_in:
      assert residual.shape == x.shape
      res2d = residual.reshape(-1, D)
      if res2d.stride(-1) != 1:
        res2d = res2d.contiguous()
    else:
      res2d = x2d

    y2d = torch.empty_like(x2d, dtype = x.dtype)
    res_out_dtype = torch.float32 if residual_in_fp32 else x.dtype
    res_out2d = torch.empty_like(x2d, dtype = res_out_dtype)
    reciprocal_stddev = torch.empty((rows,), dtype = torch.float32, device = x.device)

    block = _choose_block(D, x2d.dtype)
    num_warps = _choose_num_warps(block)
    num_stages = 2
    num_iters = triton.cdiv(D, block)
    one_pass = (num_iters == 1)

    bias_ptr = bias if has_bias else weight
    
    _rmsnorm_fwd_kernel[(rows,)](x_ptr = x2d,
                                residual_input_ptr = res2d,
                                y_ptr = y2d,
                                residual_output_ptr = res_out2d,
                                rstd_ptr = reciprocal_stddev,
                                weight_ptr = weight,
                                bias_ptr = bias_ptr,
                                row_stride = x2d.stride(0),
                                N = D,
                                eps = eps,
                                HAS_RESIDUAL_IN = has_residual_in,
                                HAS_BIAS = has_bias,
                                BLOCK = block,
                                NUM_ITERS = num_iters,
                                ONE_PASS = one_pass,
                                num_warps = num_warps,
                                num_stages = num_stages,
                                )
    
    ctx.save_for_backward(res_out2d, 
                          reciprocal_stddev, 
                          weight, 
                          bias if has_bias else torch.tensor([], device = x.device, dtype = torch.float32)
                          )
    ctx.D = D
    ctx.rows = rows
    ctx.has_residual_in = has_residual_in
    ctx.has_bias = has_bias
    ctx.prenorm = prenorm
    ctx.block = block
    ctx.num_iters = num_iters
    ctx.one_pass = one_pass
    ctx.num_warps = num_warps
    ctx.x_dtype = x.dtype

    y = y2d.reshape_as(x)
    res_out = res_out2d.reshape_as(x)
    return (y, res_out) if prenorm else y

  @staticmethod
  def backward(ctx, *grad_outputs):
    u2d, reciprocal_stddev, weight, bias_saved = ctx.saved_tensors
    D = ctx.D
    rows = ctx.rows

    dy = grad_outputs[0]
    dy2d = dy.reshape(-1, D)
  
    if dy2d.stride(-1) != 1:
      dy2d = dy2d.contiguous()

    if ctx.prenorm and len(grad_outputs) > 1 and (grad_outputs[1] is not None):
      dres_out2d = grad_outputs[1].reshape(-1, D)
      if dres_out2d.stride(-1) != 1:
        dres_out2d = dres_out2d.contiguous()
      has_dresidual_out = True
    else:
      dres_out2d = dy2d
      has_dresidual_out = False
    
    dx2d = torch.empty_like(dy2d, dtype = ctx.x_dtype)

    if ctx.has_residual_in:
      dres_in2d = torch.empty_like(dy2d, dtype = ctx.x_dtype)
    else:
      dres_in2d = dx2d
    
    has_bias = ctx.has_bias
    
    BLOCK = _choose_block(D, dy2d.dtype)
    NUM_ITERS = triton.cdiv(D, BLOCK)
    num_warps = _choose_num_warps(BLOCK)

    # For large N (ONE_PASS), use the loop kernel which is much faster/memory-efficient
    if NUM_ITERS == 1 and D <= 8192:
      # Fast path for N <= 8192: Fused kernel (One Pass)
      num_sms = torch.cuda.get_device_properties(dy.device).multi_processor_count
      target_blocks = num_sms * 2
      nrow_groups = min(target_blocks, rows)
      rows_per_program = math.ceil(rows / nrow_groups)
      
      if rows_per_program < 1: 
        rows_per_program = 1
      nrow_groups = math.ceil(rows / rows_per_program)
      
      _dw = torch.empty((nrow_groups, D), dtype = torch.float32, device = dy.device)
      _db = torch.empty((nrow_groups, D), dtype = torch.float32, device = dy.device) if has_bias else torch.empty((1,), dtype = torch.float32, device = dy.device)

      _rmsnorm_bwd_loop_rows_kernel[(nrow_groups,)](
        u_ptr = u2d,
        dy_ptr = dy2d,
        dresidual_output_ptr = dres_out2d if has_dresidual_out else dy2d,
        dx_ptr = dx2d,
        dresidual_input_ptr = dres_in2d,
        weight_ptr = weight,
        rstd_ptr = reciprocal_stddev,
        dweight_ptr = _dw,
        dbias_ptr = _db,
        stride_row = dy2d.stride(0),
        rows = rows,
        N = D,
        rows_per_program = rows_per_program,
        BLOCK = BLOCK,
        HAS_BIAS = has_bias,
        HAS_DRESIDUAL_OUT = has_dresidual_out,
        HAS_RESIDUAL_IN = ctx.has_residual_in,
        num_warps = num_warps
      )
      dweight = _dw.sum(0).to(weight.dtype)
      dbias = _db.sum(0).to(bias_saved.dtype) if has_bias else None
        
    else:
      # Compute C1 (dot products)
      c1 = torch.empty((rows,), dtype=torch.float32, device=dy.device)

      C1_BLOCK = triton.next_power_of_2(D)
      if C1_BLOCK > 16384: 
        C1_BLOCK = 16384 # Clamp
      

      _rmsnorm_bwd_c1_kernel[(rows,)](
        u_ptr=u2d, dy_ptr=dy2d, weight_ptr=weight, rstd_ptr=reciprocal_stddev, c1_ptr=c1,
        stride_row=dy2d.stride(0), N=D, eps=1e-6, # eps unused in c1 kernel
        BLOCK=C1_BLOCK
      )
      
      FINAL_BLOCK = 4096 # vibe and hand-tuned
      num_col_tiles = triton.cdiv(D, FINAL_BLOCK)
      
      num_sms = torch.cuda.get_device_properties(dy.device).multi_processor_count
      target_row_blocks = num_sms # split rows enough
      rows_per_program = math.ceil(rows / target_row_blocks)
      if rows_per_program < 1: 
        rows_per_program = 1
      num_row_tiles = math.ceil(rows / rows_per_program)
      
      # Initialize final accumulators to 0
      dweight = torch.zeros((D,), dtype=torch.float32, device=dy.device)
      dbias = torch.zeros((D,), dtype=torch.float32, device=dy.device) if has_bias else None
      
      _rmsnorm_bwd_final_kernel[(num_row_tiles, num_col_tiles)](
        u_ptr=u2d, dy_ptr=dy2d, dresidual_output_ptr=dres_out2d if has_dresidual_out else dy2d,
        dx_ptr=dx2d, dresidual_input_ptr=dres_in2d,
        weight_ptr=weight, rstd_ptr=reciprocal_stddev, c1_ptr=c1,
        dweight_ptr=dweight, dbias_ptr=dbias if has_bias else dweight, # dummy if no bias
        stride_row=dy2d.stride(0), rows=rows, N=D,
        rows_per_program=rows_per_program,
        BLOCK=FINAL_BLOCK,
        HAS_BIAS=has_bias,
        HAS_DRESIDUAL_OUT=has_dresidual_out,
        HAS_RESIDUAL_IN=ctx.has_residual_in
      )
      
      dweight = dweight.to(weight.dtype)
      if has_bias:
          dbias = dbias.to(bias_saved.dtype)

    dx = dx2d.reshape(-1, D).reshape_as(dy)
    dresidual = dres_in2d.reshape(-1, D).reshape_as(dy) if ctx.has_residual_in else None
    return dx, dweight, dbias, dresidual, None, None, None