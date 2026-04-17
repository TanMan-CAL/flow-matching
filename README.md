# Flow-matching (diffusion with velocity fields)

### Time-conditional (unconditional)
| Phase     | Step | Description |
|-----------|------|-------------|
| Training  | 1    | Sample random timestep `t ∈ [0, 1]` |
| Training  | 2    | Create interpolation `x_t = t·x₁ + (1 - t)·x₀` |
| Training  | 3    | Calculate target velocity `v = x₁ - x₀` |
| Training  | 4    | Train with MSE loss |
| Sampling  | 1    | Init noise `x₀ ~ N(0, I)` |
| Sampling  | 2    | Find `x_{t+dt} = x_t + dt · v(x_t, t)` |
| Sampling  | 3    | Return final sample `x₁` |


### Class-Conditional (with CFG)

| Phase     | Step | Description |
|-----------|------|-------------|
| Training  | 1    | Same as unconditional training |
| Training  | 2    | Add class conditioning |
| Training  | 3    | Randomly drop conditioning (10% CFG dropout) |
| Sampling  | 1    | Predict conditional velocity `v_c` |
| Sampling  | 2    | Predict unconditional velocity `v_u` |
| Sampling  | 3    | Use guidance `v = v_u + λ (v_c - v_u)` |
| Sampling  | 4    | Use guided velocity in integral |

## Model architecture
* Encoder: 3 -> 128 -> 256 -> 512 channels
* Bottleneck: Flattened latent with time/class conditioning
* Decoder: 512 -> 256 -> 128 -> 3 channels

## References
Flow Matching [Lipman et al., 2023](https://arxiv.org/abs/2210.02747);
CFG [Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598);
CIFAR-100 [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)