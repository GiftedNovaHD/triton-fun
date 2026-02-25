"""
Triton-based implementation of Discrete Ordinary Least Squares solver.
"""
import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Tuple, Optional, List, Union

from triton-fun.ops.discrete_ols_reference import lll_reduction_reference, babai_closest_vector

def _check_triton_available() -> bool:
  return torch.cuda.is_available()

@triton.jit
def _size_reduction_q_kernel(mu_matrix_ptr,
                             q_coeff_ptr,
                             row_idx,
                             num_rows,
                             mu_stride_row,
                             mu_stride_col,
                             q_stride_row,
                             BLOCK_N: tl.constexpr,
                            ):
  """
  Compute size-reduction coefficients for row k and update mu[k, :].
  """
  col_idx = tl.arange(0, BLOCK_N)
  valid_cols = col_idx < num_rows

  mu_row_k = tl.load(
    mu_matrix_ptr + row_idx * mu_stride_row + col_idx * mu_stride_col,
    mask = valid_cols,
    other = 0.0,
  )
  q_coeffs = tl.zeros([BLOCK_N], dtype = tl.float64)

  for j_offset in tl.static_range(0, BLOCK_N):
    prev_row = row_idx - 1 - j_offset
    valid_prev = prev_row >= 0
    prev_row_safe = tl.where(valid_prev, prev_row, 0)

    is_prev_col = col_idx == prev_row_safe
    mu_kj = tl.sum(tl.where(is_prev_col, mu_row_k, 0.0), axis = 0)
    mu_kj = tl.where(valid_prev, mu_kj, 0.0)
    mu_kj_abs = tl.abs(mu_kj)
    mu_kj_pos = mu_kj >= 0
    mu_kj_round = tl.where(mu_kj_pos, tl.floor(mu_kj + 0.5), -tl.floor(-mu_kj + 0.5))
    q_j = tl.where(valid_prev & (mu_kj_abs > 0.5), mu_kj_round, 0.0)

    update_cols = col_idx <= prev_row_safe
    mu_row_prev = tl.load(
      mu_matrix_ptr + prev_row_safe * mu_stride_row + col_idx * mu_stride_col,
      mask = valid_cols & update_cols & valid_prev,
      other = 0.0,
    )
    mu_row_k = tl.where(update_cols & valid_prev, mu_row_k - q_j * mu_row_prev, mu_row_k)

    write_q = (col_idx == prev_row_safe) & valid_prev
    q_coeffs = tl.where(write_q, q_j, q_coeffs)

  tl.store(mu_matrix_ptr + row_idx * mu_stride_row + col_idx * mu_stride_col,
           mu_row_k,
           mask = valid_cols,
           )
  tl.store(q_coeff_ptr + col_idx * q_stride_row, 
           q_coeffs, 
           mask = valid_cols
           )


@triton.jit
def _gs_mu_kernel_fixed(basis_ptr,
                        gs_ortho_ptr,
                        gs_norms_ptr,
                        gs_coeff_ptr,
                        row_idx,
                        num_cols,
                        basis_stride_row,
                        basis_stride_col,
                        ortho_stride_row,
                        ortho_stride_col,
                        coeff_stride_row,
                        coeff_stride_col,
                        BLOCK_J: tl.constexpr,
                        BLOCK_D: tl.constexpr
                       ):
  """
  Compute Gram-Schmidt coefficients mu[row_idx, :row_idx] for one row.
  """
  pid = tl.program_id(0)
  prev_start = pid * BLOCK_J
  prev_rows = prev_start + tl.arange(0, BLOCK_J)
  valid_prev = prev_rows < row_idx

  dot_acc = tl.zeros([BLOCK_J], dtype=tl.float64)
  col_start = 0
  while col_start < num_cols:
    col_idx = col_start + tl.arange(0, BLOCK_D)
    valid_cols = col_idx < num_cols
    basis_row = tl.load(
      basis_ptr + row_idx * basis_stride_row + col_idx * basis_stride_col,
      mask=valid_cols,
      other=0.0,
    )
    ortho_rows = tl.load(
      gs_ortho_ptr + prev_rows[:, None] * ortho_stride_row + col_idx[None, :] * ortho_stride_col,
      mask=valid_prev[:, None] & valid_cols[None, :],
      other=0.0,
    )
    dot_acc += tl.sum(ortho_rows * basis_row[None, :], axis=1)
    col_start += BLOCK_D

  prev_norms = tl.load(gs_norms_ptr + prev_rows, mask=valid_prev, other=1.0)
  mu_vals = dot_acc / (prev_norms + 1e-15)
  tl.store(gs_coeff_ptr + row_idx * coeff_stride_row + prev_rows * coeff_stride_col, mu_vals, mask=valid_prev)


@triton.jit
def _gs_q_update_kernel_fixed(basis_ptr,
                              gs_ortho_ptr,
                              gs_coeff_ptr,
                              row_idx,
                              num_cols,
                              basis_stride_row,
                              basis_stride_col,
                              ortho_stride_row,
                              ortho_stride_col,
                              coeff_stride_row,
                              coeff_stride_col,
                              BLOCK_J: tl.constexpr,
                              BLOCK_D: tl.constexpr
                             ):
  """
  Update orthogonal row Q[row_idx] = B[row_idx] - sum(mu[row_idx, j] * Q[j]).
  """
  pid = tl.program_id(0)
  col_start = pid * BLOCK_D
  col_idx = col_start + tl.arange(0, BLOCK_D)
  valid_cols = col_idx < num_cols

  basis_row = tl.load(
    basis_ptr + row_idx * basis_stride_row + col_idx * basis_stride_col,
    mask=valid_cols,
    other=0.0,
  )
  proj_acc = tl.zeros([BLOCK_D], dtype=tl.float64)

  prev_start = 0
  while prev_start < row_idx:
    prev_rows = prev_start + tl.arange(0, BLOCK_J)
    valid_prev = prev_rows < row_idx
    mu_vals = tl.load(
      gs_coeff_ptr + row_idx * coeff_stride_row + prev_rows * coeff_stride_col,
      mask=valid_prev,
      other=0.0,
    )
    ortho_rows = tl.load(
      gs_ortho_ptr + prev_rows[:, None] * ortho_stride_row + col_idx[None, :] * ortho_stride_col,
      mask=valid_prev[:, None] & valid_cols[None, :],
      other=0.0,
    )
    proj_acc += tl.sum(ortho_rows * mu_vals[:, None], axis=0)
    prev_start += BLOCK_J

  ortho_row = basis_row - proj_acc
  tl.store(
    gs_ortho_ptr + row_idx * ortho_stride_row + col_idx * ortho_stride_col,
    ortho_row,
    mask=valid_cols,
  )


@triton.jit
def _apply_q_update_kernel(basis_ptr,
                           unimod_ptr,
                           q_coeff_ptr,
                           row_idx,
                           num_rows,
                           num_cols,
                           basis_stride_row,
                           basis_stride_col,
                           unimod_stride_row,
                           unimod_stride_col,
                           q_stride_row,
                           BLOCK_J: tl.constexpr,
                           BLOCK_COL: tl.constexpr
                          ):
  """
  Apply size-reduction coefficients to basis and unimodular rows:
  B[row_idx] -= sum(q_j * B[j]); U[row_idx] -= sum(q_j * U[j]).
  """
  pid = tl.program_id(0)
  col_idx = pid * BLOCK_COL + tl.arange(0, BLOCK_COL)
  valid_basis_cols = col_idx < num_cols
  valid_unimod_cols = col_idx < num_rows

  basis_acc = tl.zeros([BLOCK_COL], dtype=tl.float64)
  unimod_acc = tl.zeros([BLOCK_COL], dtype=tl.float64)

  prev_start = 0
  while prev_start < row_idx:
    prev_rows = prev_start + tl.arange(0, BLOCK_J)
    valid_prev = prev_rows < row_idx
    q_vals = tl.load(q_coeff_ptr + prev_rows * q_stride_row, mask=valid_prev, other=0.0)

    b_vals = tl.load(
      basis_ptr + prev_rows[:, None] * basis_stride_row + col_idx[None, :] * basis_stride_col,
      mask = valid_prev[:, None] & valid_basis_cols[None, :],
      other = 0.0,
    )
    basis_acc += tl.sum(b_vals * q_vals[:, None], axis=0)

    u_vals = tl.load(
      unimod_ptr + prev_rows[:, None] * unimod_stride_row + col_idx[None, :] * unimod_stride_col,
      mask = valid_prev[:, None] & valid_unimod_cols[None, :],
      other = 0.0,
    )
    unimod_acc += tl.sum(u_vals * q_vals[:, None], axis = 0)

    prev_start += BLOCK_J

  basis_row = tl.load(basis_ptr + row_idx * basis_stride_row + col_idx * basis_stride_col,
                      mask = valid_basis_cols,
                      other = 0.0,
                      )
  tl.store(
    basis_ptr + row_idx * basis_stride_row + col_idx * basis_stride_col,
    basis_row - basis_acc,
    mask=valid_basis_cols,
  )

  unimod_row = tl.load(
    unimod_ptr + row_idx * unimod_stride_row + col_idx * unimod_stride_col,
    mask=valid_unimod_cols,
    other=0.0,
  )
  tl.store(
    unimod_ptr + row_idx * unimod_stride_row + col_idx * unimod_stride_col,
    unimod_row - unimod_acc,
    mask=valid_unimod_cols,
  )


def _update_gs_triton(basis: Tensor,
                      gs_ortho: Tensor,
                      gs_coeffs: Tensor,
                      gs_norms: Tensor,
                      row_idx: int,
                      block_prev_rows: int,
                      block_dim: int
                     ) -> None:
  """
  Update Gram-Schmidt state for a single basis row.
  """
  _, num_cols = basis.shape
  if row_idx == 0:
    gs_ortho[row_idx].copy_(basis[row_idx])
    gs_norms[row_idx] = torch.dot(gs_ortho[row_idx], gs_ortho[row_idx])
    return

  grid = (triton.cdiv(row_idx, block_prev_rows),)
  _gs_mu_kernel_fixed[grid](
    basis,
    gs_ortho,
    gs_norms,
    gs_coeffs,
    row_idx,
    num_cols,
    basis.stride(0),
    basis.stride(1),
    gs_ortho.stride(0),
    gs_ortho.stride(1),
    gs_coeffs.stride(0),
    gs_coeffs.stride(1),
    BLOCK_J=block_prev_rows,
    BLOCK_D=block_dim,
  )

  grid_q = (triton.cdiv(num_cols, block_dim),)
  _gs_q_update_kernel_fixed[grid_q](
    basis,
    gs_ortho,
    gs_coeffs,
    row_idx,
    num_cols,
    basis.stride(0),
    basis.stride(1),
    gs_ortho.stride(0),
    gs_ortho.stride(1),
    gs_coeffs.stride(0),
    gs_coeffs.stride(1),
    BLOCK_J=block_prev_rows,
    BLOCK_D=block_dim,
  )
  gs_norms[row_idx] = torch.dot(gs_ortho[row_idx], gs_ortho[row_idx])


def _size_reduction_q_triton(gs_coeffs: Tensor,
                             q_coeffs: Tensor,
                             row_idx: int,
                             block_size: int
                            ) -> None:
  """
  Launch size-reduction kernel to compute q for a single row.
  """
  num_rows = gs_coeffs.shape[0]
  grid = (1,)
  _size_reduction_q_kernel[grid](
    gs_coeffs,
    q_coeffs,
    row_idx,
    num_rows,
    gs_coeffs.stride(0),
    gs_coeffs.stride(1),
    q_coeffs.stride(0),
    BLOCK_N=block_size,
  )


def _apply_q_update_triton(basis: Tensor,
                           unimod: Tensor,
                           q_coeffs: Tensor,
                           row_idx: int
                          ) -> None:
  """
  Launch fused update kernel for basis and unimodular rows.
  """
  num_rows, num_cols = basis.shape
  grid = (triton.cdiv(max(num_cols, num_rows), 128),)
  _apply_q_update_kernel[grid](
    basis,
    unimod,
    q_coeffs,
    row_idx,
    num_rows,
    num_cols,
    basis.stride(0),
    basis.stride(1),
    unimod.stride(0),
    unimod.stride(1),
    q_coeffs.stride(0),
    BLOCK_J=32,
    BLOCK_COL=128,
  )

def lll_reduction_triton(basis: Tensor,
                         delta: float = 0.75,
                         block_j: int = 32,
                         block_d: int = 128,
                         block_n: Optional[int] = None
                        ) -> Tuple[Tensor, Tensor]:
  """
  LLL lattice basis reduction using Triton for Gram-Schmidt and size reduction.
  Falls back to the reference implementation on CPU or when CUDA is unavailable.
  """
  if not _check_triton_available() or not basis.is_cuda:
    return lll_reduction_reference(basis, delta=delta)

  num_rows, _ = basis.shape
  basis = basis.clone().to(torch.float64).contiguous()
  unimodular = torch.eye(num_rows, device=basis.device, dtype=torch.float64)

  gs_ortho = torch.zeros_like(basis)
  gs_coeffs = torch.eye(num_rows, device=basis.device, dtype=torch.float64)
  gs_norms = torch.zeros(num_rows, device=basis.device, dtype=torch.float64)

  # Initial Gram-Schmidt orthogonalization.
  for row_idx in range(num_rows):
    _update_gs_triton(basis,
                      gs_ortho,
                      gs_coeffs,
                      gs_norms,
                      row_idx,
                      block_prev_rows=block_j,
                      block_dim=block_d
                     )

  if block_n is None:
    block_n = triton.next_power_of_2(num_rows)
    if block_n > 1024:
      return lll_reduction_reference(basis, delta=delta)
  q_coeffs = torch.empty(num_rows, device=basis.device, dtype=torch.float64)

  row_idx = 1
  while row_idx < num_rows:
    _size_reduction_q_triton(gs_coeffs, q_coeffs, row_idx, block_size=block_n)
    if row_idx > 0:
      _apply_q_update_triton(basis, unimodular, q_coeffs, row_idx)

    # Lovasz condition.
    if gs_norms[row_idx] >= (delta - gs_coeffs[row_idx, row_idx - 1] ** 2) * gs_norms[row_idx - 1]:
      row_idx += 1
    else:
      # Swap basis vectors and rebuild GS rows.
      basis_swap = basis[[row_idx - 1, row_idx]].clone()
      unimod_swap = unimodular[[row_idx - 1, row_idx]].clone()
      basis[[row_idx - 1, row_idx]] = basis_swap[[1, 0]]
      unimodular[[row_idx - 1, row_idx]] = unimod_swap[[1, 0]]

      _update_gs_triton(
        basis,
        gs_ortho,
        gs_coeffs,
        gs_norms,
        row_idx - 1,
        block_prev_rows=block_j,
        block_dim=block_d,
      )
      _update_gs_triton(
        basis,
        gs_ortho,
        gs_coeffs,
        gs_norms,
        row_idx,
        block_prev_rows=block_j,
        block_dim=block_d,
      )
      row_idx = max(row_idx - 1, 1)

  return basis.to(torch.float32), unimodular.to(torch.float32)


def discrete_ols_solver_core_triton(W: Tensor,
                                    b: Optional[Tensor],
                                    y: Tensor,
                                    lll_reduction_result: Optional[Tuple[Tensor, Tensor]] = None,
                                    qr_decomposition: Optional[Tuple[Tensor, Tensor]] = None
                                   ) -> Tensor:
  """
  Triton-accelerated discrete OLS solver core. Uses Triton LLL reduction for B = W.T.
  """
  target = y - b if b is not None else y

  if lll_reduction_result is None:
    basis = W.transpose(0, 1)
    basis_reduced, unimodular = lll_reduction_triton(basis)
  else:
    basis_reduced, unimodular = lll_reduction_result

  x_reduced = babai_closest_vector(basis_reduced, target, qr_decomposition=qr_decomposition)
  x_reduced_f32 = x_reduced.to(torch.float32)
  x = torch.round(x_reduced_f32 @ unimodular).to(torch.int64)
  return x


def discrete_ols_solver_triton(W: Tensor,
                               b: Optional[Tensor],
                               y: Union[Tensor, List[Tensor]],
                               lll_reduction_result: Optional[Tuple[Tensor, Tensor]] = None,
                               qr_decomposition: Optional[Tuple[Tensor, Tensor]] = None
                              ) -> Union[Tensor, List[Tensor]]:
  """
  Wrapper for Triton-accelerated discrete OLS solver. Handles batched tensors and ragged lists.
  """
  if isinstance(y, Tensor):
    if y.dim() == 3:
      batch, seq, dim = y.shape
      y_flat = y.reshape(-1, dim)
      x_flat = discrete_ols_solver_triton(W, b, y_flat,
                                          lll_reduction_result=lll_reduction_result,
                                          qr_decomposition=qr_decomposition
                                         )
      return x_flat.reshape(batch, seq, -1)

    return discrete_ols_solver_core_triton(W, b, y,
                                           lll_reduction_result=lll_reduction_result,
                                           qr_decomposition=qr_decomposition
                                          )

  if len(y) == 0:
    return []

  lengths = [t.shape[0] for t in y]
  total_len = sum(lengths)
  if total_len == 0:
    d_in = W.shape[1]
    return [torch.zeros((0, d_in), device=y_item.device, dtype=torch.int64) for y_item in y]

  y_cat = torch.cat(y, dim=0)
  x_cat = discrete_ols_solver_core_triton(W, b, y_cat,
                                          lll_reduction_result=lll_reduction_result,
                                          qr_decomposition=qr_decomposition
                                         )
  return list(torch.split(x_cat, lengths, dim=0))
