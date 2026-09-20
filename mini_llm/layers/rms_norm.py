import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply RMS normalization.

        .. math::

            y_i = \omega_i \cdot
            \frac{x_i}{
                \sqrt{
                    \frac{1}{d}
                    \sum_{j=1}^{d} x_j^2
                    + \varepsilon
                }
            }

        where :math:`d` is the hidden size, :math:`\omega_i` is the learnable
        scaling parameter, and :math:`\varepsilon` is a small constant for
        numerical stability.
        """
        x_fp32 = x.float()
        mean_square = x_fp32.square().mean(dim=-1, keepdim=True)

        return (x_fp32 * torch.rsqrt(mean_square + self.eps) * self.weight.float()).to(
            x.dtype
        )
