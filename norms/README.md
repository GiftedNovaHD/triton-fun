# Fused Normalization operations in Triton

## Notation
Let $D \in \mathbb{N}$ denote the hidden dimension. For each token (row) index $t$ (e.g., flattening $(B, S)$ to a single index), let:
- $x^{(t)} \in \mathbb{R}^{D}$ be the input vector,
- $u^{(t)} \in \mathbb{R}^{D}$ bean optional residual input vector,
- $g \in \mathbb{R}$ be a _single_ learned scalar parameter shared channel-wise,
- $\epsilon > 0$ be constant close to 0 for numerical stability.
First define a constant scale factor:
```math
s \triangleq \sqrt{D}
```
Here, we write all formulas per-row and omit the superscript $(t)$ where unambiguous. 

## Forward Pass Operator
We define the residual output:
```math
r \triangleq
\begin{cases}
x + u, & \text{if a residual input is supplied} \\
x, & \text{otherwise.}
\end{cases}
```
This $r$ may optionally be returned (as in the case of the pre-norm architecture) and is also the value used for normalization. Next we define the clamped L2 Normalization. Recall that the unclamped L2 norm is calculated as
```math
\lVert r \rVert_{2} \triangleq \sqrt{ \sum_{i = 1}^{D} r_{i}^{2} }
```

Consider an input vector $x \in \mathbb{R}^{D}$, with $D$ denoting the hidden-dimension, an optional residual input $r_{\text{in}} \in \mathbb{R}^{D}$,
a scalar parameter $g \in \mathbb{R}$, a scaling factor $s = \sqrt{D}$, and an $\epsilon > 0$.

We define the residual as

```math
r = \begin{cases}
x + r_{\text{in}}, & \text{if residual is provided} \\
x, & \text{otherwise}
\end{cases}
```
Then define the (unclamped) L2 norm as
```math
\lVert r \rVert_{2} = \sqrt{ \sum_{i = 1}^{D} r_{i}^{2} }
```
The clamped denominator is given by $d \triangleq \max( \lVert r \rVert_{2}, \epsilon)$ which allows us to compute the inverse norm, $\mathrm{inv} \triangleq d^{-1}$. As per the SSNorm paper, the SSNorm gain is given by
```math
\gamma \triangleq s(g + 1)
```
and its output is computed by modulating
```math
y \triangleq \gamma r \mathrm{inv} = \gamma \frac{r}{\max( \lVert r \rVert_{2}, \epsilon ) }
```
Componentwise,
```math
y_{i} = \gamma \frac{r_{i}}{\max( \lVert r \rVert_{2}, \epsilon) }, \quad i = 1, \dots, D
```

### Backward Pass
Let $\mathcal{L}$ be a scalar loss. Denote the upstream gradient w.r.t. SSNorm's output by
```math
\bar{y} \triangleq \frac{\partial \mathcal{L} }{ \partial y } \in \mathbb{R}^{D}.
```
If the residual output $r$ is also returned as an auxiliary output (pre-norm) and participates in the computation graph, it _may_ receive an additional upstream gradient given by
```math
\bar{r}_{\text{aux}} \triangleq \frac{ \partial \mathcal{L}}{\partial r} \Bigg |_\text{from auxiliary output} \in \mathbb{R}^{D},
```
otherwise, $\bar{r}_{\text{aux}} = 0$.
We now move on to gradient derivation
<TBD>
