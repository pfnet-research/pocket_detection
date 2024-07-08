import logging
from dataclasses import dataclass

from torch import nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    @dataclass
    class _Config:
        pass

    @classmethod
    def get_config_class(cls):
        return cls._Config

    @classmethod
    def from_config(cls, config, *args):
        return cls(config, *args)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_metrics_keys(self):
        return self.model.get_metrics_keys()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def reconstruct(self, batch, temp=None, max_len=100):
        return self.model.reconstruct(batch, temp, max_len=max_len)

    @property
    def latent_size(self):
        return self.model.latent_size
