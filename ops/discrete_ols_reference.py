import torch
from torch import Tensor
from typing import Tuple, Optional, List, Union

def lll_reduction_reference(B: Tensor, 
                            delta: float = 0.75
                           ) -> Tuple[Tensor, Tensor]:
  """
  LLL lattice basis reduction.

  https://arxiv.org/abs/2410.22196

  Args:
    - `B` (`Tensor`): (`n`, `d`) tensor where `n` is the number of basis vectors and `d` is the dimension
    - `delta` (`float`): LLL parameter, typically in (0.25, 1). Default is 0.75

  Returns:
    - `B_reduced` (`Tensor`): (`n`, `d`) LLL-reduced basis
    - `U` (`Tensor`): (`n`, `n`) unimodular matrix such that `B_reduced = U @ B`
  """
  n, _ = B.shape
  B = B.clone().to(torch.float64)
  U = torch.eye(n, device=B.device, dtype=torch.float64)
  
  # Gram-Schmidt orthogonalization (maintained incrementally)
  Q = torch.zeros_like(B)
  mu = torch.eye(n, device=B.device, dtype=torch.float64)
  norms = torch.zeros(n, device=B.device, dtype=torch.float64)
  
  def update_gs(i):
    Q[i].copy_(B[i])
    if i > 0:
      # Vectorize the dot products: mu[i, j] = dot(B[i], Q[j]) / norms[j]
      # Use einsum for better performance: 'd,id->i' where d is dimension, i is num prev vectors
      mu_vals = torch.sum(B[i:i+1] * Q[:i], dim=1) / (norms[:i] + 1e-15)
      mu[i, :i] = mu_vals
      # Vectorize the Q update: Q[i] -= sum_j(mu[i,j] * Q[j])
      mu_vals_expanded = mu_vals.unsqueeze(0)  # (1, i)
      Q_prev = Q[:i]  # (i, d)
      Q[i] -= (mu_vals_expanded @ Q_prev).squeeze(0)
    norms[i] = torch.dot(Q[i], Q[i])

  for i in range(n):
    update_gs(i)
  
  k = 1
  while k < n:
    # Size reduction
    for j in reversed(range(k)):
      mu_kj = mu[k, j]
      if mu_kj.abs() > 0.5:
        q = torch.round(mu_kj).to(torch.int64)
        B[k] -= q * B[j]
        U[k] -= q * U[j]
        # Vectorized mu update: mu[k, :j+1] -= q * mu[j, :j+1]
        mu[k, :j+1] -= q * mu[j, :j+1]
    
    # Lovasz condition
    # norms[k] >= (delta - mu[k, k-1]**2) * norms[k-1]
    if norms[k] >= (delta - mu[k, k-1]**2) * norms[k-1]:
      k += 1
    else:
      # Swap basis vectors
      B_swap = B[[k-1, k]].clone()
      U_swap = U[[k-1, k]].clone()
      B[[k-1, k]] = B_swap[[1, 0]]
      U[[k-1, k]] = U_swap[[1, 0]]
      
      # Update GS orthogonalization for k-1 and k
      update_gs(k-1)
      update_gs(k)
      
      k = max(k - 1, 1)
      
  return B.to(torch.float32), U.to(torch.float32)

def babai_closest_vector(B_reduced: Tensor, 
                         target: Tensor, 
                         qr_decomposition: Optional[Tuple[Tensor, Tensor]] = None
                        ) -> Tensor:
  """
  Babai's nearest plane algorithm to find an approximate closest vector in the lattice.

  Args:
    - `B_reduced` (`Tensor`): (`n`, `d`) LLL-reduced basis
    - `target` (`Tensor`): (`batch_size`, `d`) or (`d`,) target vector(s)
    - `qr_decomposition` (`Optional[Tuple[Tensor, Tensor]]`): Optional precomputed (`Q`, `R`) of `B_reduced.T`

  Returns:
    - `x` (`Tensor`): (`batch_size`, `n`) or (`n`,) integer coefficients such that `x @ B_reduced` approx `target`
  """
  # Standard Babai's Nearest Plane uses Gram-Schmidt
  # For efficiency with batches, we can use the QR decomposition of B_reduced.T
  # B_reduced is (n, d). B_reduced.T is (d, n).
  # We want x @ B_reduced \approx target  =>  B_reduced.T @ x.T \approx target.T
  
  n, d = B_reduced.shape
  if target.dim() == 1:
    target = target.unsqueeze(0)
  
  # Use QR decomposition of B_reduced.T: Q (d, n), R (n, n)
  # Convert to float64 early for better numerical stability
  if qr_decomposition is not None:
    Q, R = qr_decomposition
  else:
    B_reduced_f64 = B_reduced.to(torch.float64)
    Q, R = torch.linalg.qr(B_reduced_f64.T, mode='reduced')
  
  # target.T approx Q @ R @ x.T
  # Q.T @ target.T approx R @ x.T
  
  # target_proj: (n, batch_size)
  target_f64 = target.to(torch.float64)
  target_proj = Q.T @ target_f64.T  # (n, batch_size)
  
  # Solve R @ x.T = target_proj for integer x
  # Since R is upper triangular, use solve_triangular
  # This is faster than using manual back-substitution
  if n > d:
    raise ValueError(f"Underdetermined lattice problem (n={n} > d={d}). Index recovery may be non-unique.")
  x_cont = torch.linalg.solve_triangular(R, target_proj, upper=True, unitriangular=False)
  
  x = torch.round(x_cont).to(torch.int64)
      
  return x.T.to(torch.int64)

def discrete_ols_solver_core_reference(W: Tensor, 
                                       b: Optional[Tensor], 
                                       y: Tensor, 
                                       lll_reduction_result: Optional[Tuple[Tensor, Tensor]] = None,
                                       qr_decomposition: Optional[Tuple[Tensor, Tensor]] = None
                                      ) -> Tensor:
  """
  Solve the discrete OLS problem: find vector of integers x such that ||x @ W.T + b - y||^2 is minimized.
  This is equivalent to finding vector of integers x such that x @ W.T approx y - b.

  Args:
    - `W` (`Tensor`): (`d_out`, `d_in`) weight matrix of the projection layer
    - `b` (`Optional[Tensor]`): (`d_out`,) bias vector of the projection layer
    - `y` (`Tensor`): (`batch_size`, `d_out`) target vectors (quantized latent states)
    - `lll_reduction_result` (`Optional[Tuple[Tensor, Tensor]]`): Optional precomputed (`B_reduced`, `U`) for `B = W.T`
    - `qr_decomposition` (`Optional[Tuple[Tensor, Tensor]]`): Optional precomputed (`Q`, `R`) for `B_reduced.T`

  Returns:
    - `x` (`Tensor`): (`batch_size`, `d_in`) batch of vector of integers
  """
  # Target t = y - b
  if b is not None:
    t = y - b
  else:
    t = y
      
  # Basis vectors are the rows of W (which are the columns of W.T)
  # W is (d_out, d_in). W.T is (d_in, d_out).
  # x @ W.T is (batch_size, d_out).
  # So basis B is W.T, which has shape (d_in, d_out).
  
  if lll_reduction_result is None:
    B = W.transpose(0, 1)
    B_reduced, U = lll_reduction_reference(B)
  else:
    B_reduced, U = lll_reduction_result
  
  # 2. Babai's closest vector in the reduced basis
  x_reduced = babai_closest_vector(B_reduced, t, qr_decomposition=qr_decomposition)
  
  # 3. Transform back to original basis
  # x = x_reduced @ U
  x_reduced_f32 = x_reduced.to(torch.float32)
  x = torch.round(x_reduced_f32 @ U).to(torch.int64)
  
  return x

def discrete_ols_solver_reference(W: Tensor, 
                                  b: Optional[Tensor], 
                                  y: Union[Tensor, List[Tensor]],
                                  lll_reduction_result: Optional[Tuple[Tensor, Tensor]] = None,
                                  qr_decomposition: Optional[Tuple[Tensor, Tensor]] = None
                                 ) -> Union[Tensor, List[Tensor]]:
  """
  Wrapper for discrete_ols_solver that handles both batched tensors and ragged lists of tensors.

  Args:
    - `W` (`Tensor`): (`d_out`, `d_in`) weight matrix of the projection layer
    - `b` (`Optional[Tensor]`): (`d_out`,) bias vector of the projection layer
    - `y` (`Union[Tensor, List[Tensor]]`): (`batch_size`, `d_out`) target vectors (quantized latent states)
    - `lll_reduction_result` (`Optional[Tuple[Tensor, Tensor]]`): Optional precomputed (`B_reduced`, `U`) for `B = W.T`
    - `qr_decomposition` (`Optional[Tuple[Tensor, Tensor]]`): Optional precomputed (`Q`, `R`) for `B_reduced.T`

  Returns:
    - `x` (`Union[Tensor, List[Tensor]]`): (`batch_size`, `d_in`) batch of vector of integers
  """
  if isinstance(y, Tensor):
    # Handle 3D (batch, seq, dim) by flattening to 2D
    if y.dim() == 3:
      batch, seq, dim = y.shape
      y_flat = y.reshape(-1, dim)
      x_flat = discrete_ols_solver_reference(W, b, y_flat, 
                                             lll_reduction_result=lll_reduction_result, 
                                             qr_decomposition=qr_decomposition
                                            )
      return x_flat.reshape(batch, seq, -1)
    
    return discrete_ols_solver_core_reference(W, b, y, 
                                              lll_reduction_result=lll_reduction_result, 
                                              qr_decomposition=qr_decomposition
                                             )

  # Handle List[Tensor] (ragged list)
  if len(y) == 0:
    return []

  lengths = [t.shape[0] for t in y]
  total_len = sum(lengths)
  
  if total_len == 0:
    d_in = W.shape[1]
    return [torch.zeros((0, d_in), device=y_item.device, dtype=torch.int64) for y_item in y]

  y_cat = torch.cat(y, dim=0)
  x_cat = discrete_ols_solver_core_reference(W, b, y_cat, 
                                             lll_reduction_result=lll_reduction_result, 
                                             qr_decomposition=qr_decomposition
                                            )
  
  return list(torch.split(x_cat, lengths, dim=0))