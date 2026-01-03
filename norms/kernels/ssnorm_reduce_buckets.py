import triton
import triton.language as tl

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