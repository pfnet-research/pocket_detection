import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from pokeformer.config_wrapper import ConfigWrapper, load_class
from pokeformer.datasets.dataset_builder_factory import DatasetBuilderFactory
from pokeformer.models.predictor_factory import PredictorFactory
from pokeformer.utils import setup_random_seed
from pokeformer.utils.logger_config import LoggerConfig, apply_logger_config

logger = logging.getLogger(__name__)


@dataclass
class MainConfig:
    dataset_builder: Any
    predictor: Any
    sampler: Any
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    seed: int = 0
    dump_config_path: Optional[str] = None


class SamplerFactory:
    @staticmethod
    def get_class(class_name):
        return load_class(class_name)


def main():

    # config
    myconf = ConfigWrapper.from_cli(MainConfig)
    config = myconf.dict_config

    apply_logger_config(config.logger)
    setup_random_seed(config.seed)
    logger.debug(f"set random seed: {config.seed}")

    #####

    dsbuilder = myconf.create_obj(DatasetBuilderFactory, "dataset_builder")
    dataset = dsbuilder.create_dataset()
    predictor = myconf.create_obj(
        PredictorFactory, "predictor", dataset.get_vocab_dict()
    )

    infer = myconf.create_obj(SamplerFactory, "sampler")
    infer.run(predictor, dataset)


if __name__ == "__main__":
    main()
    logger.info("DONE")
