# Triton fundamental
JX learns and plays around with Triton! 

### Current Implementations
- Funky arbitrary block-scaled matrix multiplication on Hopper
  - Tested on H100 and H200
- Single-Scale RMSNorm from [Outlier-Safe Pre-Training](https://arxiv.org/pdf/2506.19697) as a fused kernel
  - Consistent $~1.6\times$ speed-up over Mamba2's Triton RMSNorm implementation
  - More than $3\times$ speed-up over default Torch implementation (including `torch.compile` versions)
- First attempt at standard fused residual RMSNorm 
 
### Stuff to try in future
- Block-scaled matmul to take advantage of Blackwell's `tcgen05.mma` instruction
- Other DSLs e.g. [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/) once it is supported on Hopper, and TileLang
