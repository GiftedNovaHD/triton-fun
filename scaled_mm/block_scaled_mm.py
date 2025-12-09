import torch
import triton
import triton.language as tl
from triton import Config, autotune

@autotune(
  configs=[
    Config({}, num_warps=2, num_stages=2),
    Config({}, num_warps=2, num_stages=3),
    Config({}, num_warps=4, num_stages=2),
    Config({}, num_warps=4, num_stages=3),
    Config({}, num_warps=8, num_stages=2),
    Config({}, num_warps=8, num_stages=3),
    Config({}, num_warps=16, num_stages=2),
    Config({}, num_warps=16, num_stages=3),
  ],
  key=["M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K"],
)

@triton.jit
def block_scaled_mm_kernel(
  A_ptr, B_ptr, C_ptr,
  Sa_ptr, Sb_ptr,       # per-tile scales
  M, N, K, 
  stride_am, stride_ak,
  stride_bk, stride_bn, 
  stride_cm, stride_cn,
  M_blocks, N_blocks, K_blocks,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_K: tl.constexpr,
  ):
  """
  Triton kernel for block-scaled matrix multiplication.
  Computes C = (A * Sa) @ (B * Sb) where Sa and Sb are block-wise scales.
  """
  # 1. Program ID maps to the block in the output matrix C
  pid_m = tl.program_id(0) # tile index in M dimension
  pid_n = tl.program_id(1) # tile index in N dimension

  # 2. Calculate starting offsets for the blocks in M and N dimensions
  offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  
  # 3. Create masks to handle boundary conditions (if dimensions aren't multiples of block size)
  mask_m = offset_m < M
  mask_n = offset_n < N

  # 4. Initialize accumulator for the dot product
  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

  # 5. Loop over the K dimension in chunks of BLOCK_K
  for k_block in tl.range(0, K_blocks):
    k0 = k_block * BLOCK_K 
    offset_k = k0 + tl.arange(0, BLOCK_K)
    mask_k = offset_k < K

    # More compiler optimization
    tl.multiple_of(offset_m, BLOCK_M)
    tl.multiple_of(offset_n, BLOCK_N)
    tl.multiple_of(offset_k, BLOCK_K)

    tl.max_contiguous(offset_m, BLOCK_M)
    tl.max_contiguous(offset_n, BLOCK_N)
    tl.max_contiguous(offset_k, BLOCK_K)

    # 6. Load tile from Matrix A [BLOCK_M, BLOCK_K]
    # Formula: ptr + (row_idx * row_stride) + (col_idx * col_stride)
    a = tl.load(
      A_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak,
      mask = mask_m[:, None] & mask_k[None, :],
      other = 0.0,
    )

    # 7. Load tile from Matrix B [BLOCK_K, BLOCK_N]
    b = tl.load(
      B_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn,
      mask = mask_k[:, None] & mask_n[None, :],
      other = 0.0,
    )

    # 8. Load Per-tile scales in row-major fashion
    # Sa corresponds to the current block of A: shape [M_blocks, K_blocks]
    # Sb corresponds to the current block of B: shape [K_blocks, N_blocks]
    # These are scalar values for the entire block
    sa = tl.load(Sa_ptr + pid_m * K_blocks + k_block)
    sb = tl.load(Sb_ptr + k_block * N_blocks + pid_n)

    scale = sa * sb
    acc += tl.dot(a, b, out_dtype=tl.float32) * scale 

    # dot = tl.dot(a, b, out_dtype=tl.float32)
    # scale = sa * sb
    # acc = tl.math.fma(dot, scale, acc)
    # FMA does not seem to provide a noticeable speed up. 

  # 10. Store result into Matrix C
  # The loop is finished, so we write the accumulated block to memory
  tl.store(
    C_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn,
    acc, 
    mask = mask_m[:, None] & mask_n[None, :],
  )

def block_scaled_mm(
  a: torch.Tensor, 
  b: torch.Tensor,
  scale_a: torch.Tensor, # [M_blocks, K_blocks]
  scale_b: torch.Tensor, # [K_blocks, N_blocks]
  BLOCK_M=128,
  BLOCK_N=128,
  BLOCK_K=16,
  out_dtype=torch.float16
  ): 
  """
  Entry point for block-scaled matrix multiplication.
  
  Args:
    a: Input matrix A [M, K]
    b: Input matrix B [K, N]
    scale_a: Scales for blocks of A [M_blocks, K_blocks]
    scale_b: Scales for blocks of B [K_blocks, N_blocks]
    BLOCK_M: Block size for M dimension
    BLOCK_N: Block size for N dimension
    BLOCK_K: Block size for K dimension
  """
  # Check dimensions
  assert a.ndim == 2 and b.ndim == 2, "block_scaled_mm currently only supports 2D (M,K) @ (K,N) matmul"
  M, K = a.shape
  K2, N = b.shape
  assert K == K2, f"Dimension mismatch: A={a.shape}, B={b.shape}"

  device = a.device
  
  # Allow float16, bfloat16, float32. Default to float32 if not specified or if input is float32.
  # But for this kernel, we want to keep the input precision if possible to save bandwidth.
  # The kernel loads them as-is.
  
  # We enforce contiguous memory layout.
  a = a.contiguous()
  b = b.contiguous()

  # Calculate the number of blocks along each dimension
  M_blocks = (M + BLOCK_M - 1) // BLOCK_M
  N_blocks = (N + BLOCK_N - 1) // BLOCK_N
  K_blocks = (K + BLOCK_K - 1) // BLOCK_K

  # Prepare scales: must be on correct device and contiguous
  scale_dtype = torch.float32
  scale_a = scale_a.to(device=device, dtype=scale_dtype).contiguous()
  scale_b = scale_b.to(device=device, dtype=scale_dtype).contiguous()
  
  # Validate scale shapes
  expected_scale_a = (M_blocks, K_blocks)
  expected_scale_b = (K_blocks, N_blocks)
  assert scale_a.shape == expected_scale_a, f"scale_a shape mismatch: got {scale_a.shape}, expected {expected_scale_a}"
  assert scale_b.shape == expected_scale_b, f"scale_b shape mismatch: got {scale_b.shape}, expected {expected_scale_b}"

  # Allocate output tensor c
  c = torch.empty((M, N), device=device, dtype=torch.float32)

  # Define the grid size for the kernel launch
  # One program instance for each [BLOCK_M, BLOCK_N] tile in the output
  grid = (M_blocks, N_blocks)

  # Launch the Triton kernel
  block_scaled_mm_kernel[grid](
    a, b, c,                  # Pointers to matrices
    scale_a, scale_b,         # Pointers to scales
    M, N, K,                  # Matrix dimensions
    a.stride(0), a.stride(1), # Strides for A
    b.stride(0), b.stride(1), # Strides for B
    c.stride(0), c.stride(1), # Strides for C
    M_blocks, N_blocks, K_blocks, # Block counts
    BLOCK_M=BLOCK_M,          # Compile-time constants
    BLOCK_N=BLOCK_N,
    BLOCK_K=BLOCK_K,
    # num_warps=4,              # Kernel configuration
    # num_stages=2
  )

  return c.to(out_dtype) if out_dtype != torch.float32 else c
