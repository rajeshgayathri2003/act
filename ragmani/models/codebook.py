"""Scalars as two-hot codes over fixed bins, as DreamerV3 predicts rewards and values.

TwoHotBins holds the bins: the two-hot target, the cross-entropy loss, and decoding logits to the expected bin centre.
TwoHotHead maps features straight to bin logits, as DreamerV3's heads do. TwoHotCodebook keeps one table of bin embeddings
that works both ways: encode turns a scalar into an embedding and logits scores an embedding against every bin.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def symlog(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs() / scale)


def symexp(y: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sign(y) * torch.expm1(y.abs()) * scale


class TwoHotBins(nn.Module):
    """Scalars in [low, high] over num_bins evenly spaced centres, in symlog space when symlog is set; values outside clamp to the ends."""

    def __init__(self, num_bins: int, low: float, high: float, symlog: bool = False, scale: float = 1.0):
        super().__init__()
        assert low < high and num_bins >= 2
        self.symlog, self.scale = symlog, scale
        self.register_buffer("centers", torch.linspace(self.transform(torch.tensor(low)).item(), self.transform(torch.tensor(high)).item(), num_bins), persistent=False)
        self.step = (self.centers[-1] - self.centers[0]).item() / (num_bins - 1)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return symlog(x, self.scale) if self.symlog else x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return symexp(y, self.scale) if self.symlog else y

    def two_hot(self, x: torch.Tensor) -> torch.Tensor:
        """(...) -> (..., num_bins): weight split between the two centres either side of x, so rows sum to one."""
        num_bins = self.centers.numel()
        y = self.transform(x).clamp(self.centers[0], self.centers[-1])
        pos = ((y - self.centers[0]) / self.step).clamp(0.0, num_bins - 1)
        lo = pos.floor().long().clamp(max=num_bins - 2)
        w_hi = (pos - lo.to(pos.dtype)).clamp(0.0, 1.0)
        code = x.new_zeros(*x.shape, num_bins)
        code.scatter_(-1, lo[..., None], (1.0 - w_hi)[..., None])
        code.scatter_add_(-1, (lo + 1)[..., None], w_hi[..., None])
        return code

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """(..., num_bins) -> (...): the expected centre under softmax(logits), mapped back out of symlog space."""
        return self.inverse(logits.softmax(dim=-1) @ self.centers)

    def loss(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(..., num_bins) and (...) -> (...): cross-entropy of the predicted bins against x's two-hot code."""
        return -(self.two_hot(x) * F.log_softmax(logits, dim=-1)).sum(dim=-1)


class TwoHotHead(TwoHotBins):
    """Features -> bin logits for each of outputs scalars through one linear layer, decoded to their expected centres."""

    def __init__(self, in_dim: int, outputs: int, num_bins: int, low: float, high: float, symlog: bool = False, scale: float = 1.0):
        super().__init__(num_bins, low, high, symlog=symlog, scale=scale)
        self.outputs = outputs
        self.proj = nn.Linear(in_dim, outputs * num_bins)
        nn.init.zeros_(self.proj.weight)  # uniform bins at the start, as DreamerV3 starts its reward and critic heads
        nn.init.zeros_(self.proj.bias)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """(..., in_dim) -> (..., outputs, num_bins)."""
        return self.proj(h).unflatten(-1, (self.outputs, self.centers.numel()))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(..., in_dim) -> the scalars (..., outputs) and the logits (..., outputs, num_bins) they were decoded from."""
        logits = self.logits(h)
        return self.decode(logits), logits


class TwoHotCodebook(TwoHotBins):
    """TwoHotBins with an embedding per bin, shared by encode (scalar -> embedding) and logits (embedding -> bin scores)."""

    def __init__(self, embedding_dim: int, num_bins: int, low: float, high: float, symlog: bool = False, scale: float = 1.0, norm_eps: float = 1e-5):
        super().__init__(num_bins, low, high, symlog=symlog, scale=scale)
        self.codes = nn.Embedding(num_bins, embedding_dim)
        self.norm = nn.RMSNorm(embedding_dim, eps=norm_eps)
        self.embedding_dim = embedding_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(...) -> (..., embedding_dim)."""
        return self.norm(self.two_hot(x) @ self.codes.weight)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """(..., embedding_dim) -> (..., num_bins): h scored against each bin's embedding, the same table encode reads."""
        return h @ self.codes.weight.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class RewardCodebook(TwoHotCodebook):
    """Scalar reward -> embedding in symlog space. An odd bin count puts zero on a centre."""

    def __init__(self, embedding_dim: int, num_bins: int = 33, max_reward: float = 20.0, min_reward: float = 0.1, norm_eps: float = 1e-5):
        assert num_bins % 2 == 1, "num_bins must be odd so that zero is a bin centre"
        super().__init__(embedding_dim, num_bins, -max_reward, max_reward, symlog=True, scale=min_reward, norm_eps=norm_eps)
