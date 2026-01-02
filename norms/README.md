# Fused Normalization operations in Triton
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
\parallel r \parallel_{2} = \sqrt{ \sum_{i = 1}^{D} r_{i}^{2} }
```

