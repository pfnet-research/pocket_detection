import logging
from dataclasses import dataclass
from typing import Any

from pokeformer.config_wrapper import load_class, obj_from_config

logger = logging.getLogger(__name__)


@dataclass
class GenericDatasetBuilderConfig:
    scratch: Any
    dataset: Any


class GenericDatasetBuilder:
    @classmethod
    def get_config_class(cls):
        return GenericDatasetBuilderConfig

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def __init__(self, config):
        self.config = config

    def create_dataset(self):
        config = self.config
        obj, res_cfg = obj_from_config(load_class, config["dataset"])
        self.result_config = res_cfg
        return obj
