import logging
from math import pi as PI

import torch
from torch import nn

logger = logging.getLogger(__name__)


class GaussianEmbedding(torch.nn.Module):
    def __init__(
        self, out_dim=50, dist_resoln=None, min_x=0.0, max_x=5.0, cos_dump=False
    ):
        super().__init__()

        num_div = None
        if dist_resoln is not None:
            num_div = int((max_x - min_x) / dist_resoln)
            logger.info(f"{num_div=}")
        if num_div is None:
            num_basis = out_dim
        else:
            num_basis = num_div

        offset = torch.linspace(min_x, max_x, num_basis)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)
        self.cos_dump = cos_dump
        self.start = min_x
        self.stop = max_x

        if num_div is None:
            self.linear = None
        else:
            self.linear = nn.Linear(num_div, out_dim)

    def forward(self, dist):
        delta = dist.view(-1, 1) - self.offset.view(1, -1)
        result = torch.exp(self.coeff * torch.pow(delta, 2))

        if self.cos_dump:
            C = torch.where(
                dist < self.start,
                0,
                (dist - self.start) * PI / (self.stop - self.start),
            )
            C = 0.5 * (torch.cos(C) + 1.0)
            result = result * C.view(-1, 1)

        if self.linear is not None:
            result = self.linear(result)
        return result
