"""
Triton-based block-scaled matrix multiplication using Hopper Tensor Memory Accelerator (TMA).

This module defines:
- A persistent, autotuned Triton kernel `block_scaled_mm_kernel_tma_persistent` that multiplies
  A[M,K] and B[K,N] with per-block scales Sa[M_blocks,K_blocks] and Sb[K_blocks,N_blocks],
  accumulating in fp32 and writing to C[M,N]. It uses device-side TMA descriptors to stream
  tiles from global memory efficiently and supports optional warp specialization and a
  sub-tile epilogue for improved store coalescing.
- A Python wrapper `block_scaled_mm_tma` that validates inputs, prepares scales, and launches
  the kernel with a persistent grid sized to the number of SMs.

All dimensions must be divisible by their respective BLOCK_* sizes for the TMA path.
"""
import torch
import triton
import triton.language as tl
from triton import Config, autotune

_ALLOC_SET: bool = False
def _ensure_triton_allocator() -> None:
  """
  Install a minimal CUDA allocator for Triton TMA descriptors/buffers.

  Triton can request temporary buffers on the device when building or using
  tensor memory descriptors. We redirect allocations to `torch.empty` on the
  active CUDA device to avoid host-side allocations and ensure correct device
  placement. This is a no-op if already installed.
  """
  global _ALLOC_SET
  if _ALLOC_SET: 
    return
  def alloc_fn(size: int, alignment: int, stream=None):
    return torch.empty(size, device="cuda", dtype=torch.int8)
  triton.set_allocator(alloc_fn)
  _ALLOC_SET = True

_ensure_triton_allocator()

@autotune(
  configs=[
    # === Standard Configs (Persistent + Warp Specialize) ===
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=5),
    
    # === Optimized for Large Tiles (BK=64) ===
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": True, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=5),
    
    # === Optimized for Small Tiles (BK=32) ===
    # Attempt 1: Deeper pipeline to hide latency
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=8),
    
    # Attempt 2: Larger L2 swizzle (GROUP_M) to reduce DRAM pressure
    Config({"GROUP_M": 32, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),
    Config({"GROUP_M": 32, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=6),
    
    # Attempt 3: Fewer warps (reduced synchronization overhead)
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=6),
    
    # Attempt 4: Non-Warp-Specialized Persistent (sometimes faster for small tiles)
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=4),
    Config({"GROUP_M": 8, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=4),
    Config({"GROUP_M": 32, "EPILOGUE_SUBTILE": False, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=4),
  ],
  key=["M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K"],
)
@triton.jit
def block_scaled_mm_kernel_tma_persistent(
  A_ptr, B_ptr, C_ptr,
  Sa_ptr, Sb_ptr,
  M, N, K,
  stride_am, stride_ak,
  stride_bk, stride_bn,
  stride_cm, stride_cn,
  NUM_SMS: tl.constexpr,
  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
  GROUP_M: tl.constexpr, 
  EPILOGUE_SUBTILE: tl.constexpr,
  WARP_SPECIALIZE: tl.constexpr,
):
  """
  Persistent TMA kernel for block-scaled GEMM on Hopper-class GPUs.

  Parameters
  ----------
  `A_ptr`, `B_ptr`, `C_ptr`:
    Pointers to A[M,K], B[K,N], and C[M,N] in device memory.
  `Sa_ptr`, `Sb_ptr`:
    Pointers to per-block scales laid out as:
      `Sa_ptr`: [M_blocks, K_blocks] flattened row-major
      `Sb_ptr`: [K_blocks, N_blocks] flattened row-major
  M, N, K:
    `M`, `N`, `K`: Problem sizes.
  `stride_*`:
    Row/column strides for A, B, and C in elements.
  `NUM_SMS`:
    Number of Streaming Multiprocessors (SMs), used to size the persistent grid.
  `BLOCK_M`, `BLOCK_N`, `BLOCK_K`:
    Tile shapes for TMA transfers and computation.
  `GROUP_M`:
    L2 swizzle span in the `M` dimension to improve cache locality.
  `EPILOGUE_SUBTILE`:
    If True, stores are split into two sub-tiles to improve store coalescing.
  `WARP_SPECIALIZE`:
    If True, enables warp-specialized pipelining of load/compute on Hopper.
  """
  dtype_out = C_ptr.dtype.element_ty

  # Program id along axis 0 identifies the "persistent program" running on an SM.
  # Instead of launching one program per output tile, we launch up to NUM_SMS
  # programs and let each iterate over many tiles. This reduces launch/scheduling
  # overhead and improves L2 locality for iterative accesses.
  start_pid = tl.program_id(axis=0)
  
  # Number of tiles in the output along M and N
  num_pid_m = tl.cdiv(M, BLOCK_M)
  num_pid_n = tl.cdiv(N, BLOCK_N)
  
  # Total number of K tiles we will iterate over for each output tile
  k_tiles = tl.cdiv(K, BLOCK_K)
  num_tiles = num_pid_m * num_pid_n

  # Device-side TMA descriptors: describe tiled 2D tensor shapes/strides for A, B, and C.
  # The `block_shape` declares the transfer tile shape used by the hardware-assisted
  # 2D memory operation. These should match the compute tile sizes to avoid redundant
  # data movement and simplify indexing.
  a_desc = tl.make_tensor_descriptor(
    A_ptr,
    shape=[M, K],
    strides=[stride_am, stride_ak],
    block_shape=[BLOCK_M, BLOCK_K],
  )
  b_desc = tl.make_tensor_descriptor(
    B_ptr,
    shape=[K, N],
    strides=[stride_bk, stride_bn],
    block_shape=[BLOCK_K, BLOCK_N],
  )
  c_desc = tl.make_tensor_descriptor(
    C_ptr,
    shape=[M, N],
    strides=[stride_cm, stride_cn],
    block_shape=[BLOCK_M, BLOCK_N // 2 if EPILOGUE_SUBTILE else BLOCK_N],
  )

  # Disable loop flattening when warp specialization is enabled to preserve
  # producer/consumer pipelining between warps on Hopper.
  FLATTEN_K = False if WARP_SPECIALIZE else True

  # L2 swizzle: group program ids along M to improve L2 reuse across N
  num_pid_in_group = GROUP_M * num_pid_n

  # Persistent loop:
  # Step through the logical tile ids in strides of NUM_SMS such that each
  # persistent program (one per SM) is responsible for a disjoint subset of tiles.
  for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
    # Map a linear tile id to grouped (pid_m, pid_n) coordinates. GROUP_M swizzles
    # tiles along M to keep recently used rows of A resident in L2 across columns of B.
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    # Handle partial tail group when num_pid_m is not a multiple of GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id - group_id * num_pid_in_group

    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # Compute the row/column offsets for this output tile
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    # Accumulators live in registers and accumulate in fp32 for numerical headroom
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in units of BLOCK_K. With warp specialization enabled,
    # Triton pipelines producer (load) and consumer (matmul) warps to overlap
    # memory and compute. FLATTEN_K=False preserves this pipeline structure.
    for kb in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE, flatten=FLATTEN_K):
      offs_k = kb * BLOCK_K

      # Hardware-assisted 2D transfers bring an (BLOCK_M x BLOCK_K) tile of A and
      # a (BLOCK_K x BLOCK_N) tile of B into registers for this step.
      a = a_desc.load([offs_am, offs_k])
      b = b_desc.load([offs_k, offs_bn])

      # Per-block scales:
      #  - sa indexes Sa[pid_m, kb] where pid_m in [0, M_blocks), kb in [0, K_blocks)
      #  - sb indexes Sb[kb, pid_n] where pid_n in [0, N_blocks)
      # The linearization is: idx = row * num_cols + col.
      sa = tl.load(Sa_ptr + pid_m * k_tiles + kb).to(tl.float32)
      sb = tl.load(Sb_ptr + kb * num_pid_n + pid_n).to(tl.float32)

      # Apply block-wise scaling to the contribution from this K tile.
      # tl.dot uses out_dtype to control the accumulator precision of the dot.
      scale = sa * sb
      acc += tl.dot(a, b, out_dtype=tl.float32) * scale

    if EPILOGUE_SUBTILE: 
      # Epilogue sub-tiling splits the N dimension into two halves to improve
      # store coalescing (fewer stride conflicts) on some shapes/hardware.
      # We reshape (M, N) -> (M, 2, N/2), swap the last two axes so that each
      # half-N tile is contiguous, then split the last dim into two tensors.
      acc_reshaped = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
      acc_reshaped = tl.permute(acc_reshaped, (0, 2, 1))
      acc0, acc1 = tl.split(acc_reshaped)

      c_desc.store([offs_am, offs_bn], acc0.to(dtype_out))
      c_desc.store([offs_am, offs_bn + (BLOCK_N // 2)], acc1.to(dtype_out))
    else: 
      c_desc.store([offs_am, offs_bn], acc.to(dtype_out))

def block_scaled_mm_tma(
  a: torch.Tensor, b: torch.Tensor,
  scale_a: torch.Tensor,  # [M_blocks, K_blocks]
  scale_b: torch.Tensor,  # [K_blocks, N_blocks]
  BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 64,
  out_dtype: torch.dtype | None = None,
  warp_specialize: bool = True, # ignored, controlled by autotune
  flatten: bool = True, # ignored
)-> torch.Tensor:
  """
  Block-scaled GEMM using a Triton TMA kernel (persistent grid).

  Expects all dimensions to be divisible by BLOCK_* to use the TMA path.
  Moves `scale_a` and `scale_b` to the correct device/dtype, validates shapes,
  allocates the output, and launches the autotuned persistent kernel.

  Args:
    `a`: `A[M, K]` on CUDA.
    `b`: `B[K, N]` on CUDA.
    `scale_a`: `Sa[M_blocks, K_blocks]` in fp32.
    `scale_b`: `Sb[K_blocks, N_blocks]` in fp32.
    `BLOCK_M`, `BLOCK_N`, `BLOCK_K`: tile sizes.
    `out_dtype`: output dtype; defaults to `a.dtype` if half-like otherwise fp16.

  Returns:
    `C[M, N]` with per-block scaling applied to the product `A @ B`.
  """
  _ensure_triton_allocator()

  assert a.is_cuda and b.is_cuda

  # Expect 2D matrices (no implicit batching in this path).
  # For batched GEMM, prefer a higher-level loop or batched kernel.
  assert a.ndim == 2 and b.ndim == 2
  M, K = a.shape
  K2, N = b.shape
  assert K == K2, f"Dimension mismatch: A={a.shape}, B={b.shape}"

  # TMA requires shapes to be block-aligned so that hardware 2D transfers map
  # cleanly to tiles. If not divisible, either pad or use the non-TMA kernel.
  assert M % BLOCK_M == 0, "TMA path requires M divisible by BLOCK_M (pad or use non-TMA kernel)."
  assert N % BLOCK_N == 0, "TMA path requires N divisible by BLOCK_N (pad or use non-TMA kernel)."
  assert K % BLOCK_K == 0, "TMA path requires K divisible by BLOCK_K (pad or use non-TMA kernel)."
  
  # Contiguity guarantees simple row-major strides for descriptors and improves
  # memory coalescing. Irregular strides can degrade TMA efficiency.
  a = a.contiguous()
  b = b.contiguous()

  # Compute the number of tiles per dimension. With exact divisibility, these
  # equal M // BLOCK_M, N // BLOCK_N, K // BLOCK_K.
  M_blocks = (M + BLOCK_M - 1) // BLOCK_M
  N_blocks = (N + BLOCK_N - 1) // BLOCK_N
  K_blocks = (K + BLOCK_K - 1) // BLOCK_K

  # Scales are applied in fp32 for numerical stability. Move once to correct
  # device/dtype to avoid repeated casts inside the kernel.
  if scale_a.device != a.device or scale_a.dtype != torch.float32:
    scale_a = scale_a.to(device=a.device, dtype=torch.float32)

  if scale_b.device != b.device or scale_b.dtype != torch.float32:
    scale_b = scale_b.to(device=b.device, dtype=torch.float32)

  # Ensure row-major, linear layout so that index = row * num_cols + col holds
  scale_a, scale_b = scale_a.contiguous(), scale_b.contiguous()

  # There should be exactly one scale per logical tile
  assert scale_a.shape == (M_blocks, K_blocks), f"scale_a {scale_a.shape} vs {(M_blocks, K_blocks)}"
  assert scale_b.shape == (K_blocks, N_blocks), f"scale_b {scale_b.shape} vs {(K_blocks, N_blocks)}"

  # Default to a half-like output when possible; accumulation remains fp32
  if out_dtype is None:
    out_dtype = a.dtype if a.dtype in (torch.float16, torch.bfloat16) else torch.float16

  # Allocate output without initialization for performance
  c = torch.empty((M, N), device=a.device, dtype=out_dtype)

  # Persistent launch configuration: one long-lived program per SM iterates
  # over many tiles to reduce launch overhead and improve cache locality
  num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
  num_tiles = M_blocks * N_blocks

  # Persistent grid: launch at most one program per SM to iterate over tiles.
  # Each persistent program steps through tiles by NUM_SMS to cover the full grid.
  grid = (min(num_tiles, num_sms), 1, 1)

  # Launch kernel with element-wise strides. NUM_SMS and BLOCK_* are constexprs
  # that allow Triton to specialize the code path (and autotune across configs)
  # Note: warp specialization and epilogue behavior are controlled by autotune
  block_scaled_mm_kernel_tma_persistent[grid](
    a, b, c,
    scale_a, scale_b, 
    M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    NUM_SMS=num_sms,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
  )
  return c
