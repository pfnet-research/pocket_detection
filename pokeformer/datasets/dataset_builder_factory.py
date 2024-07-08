import logging

from pokeformer.config_wrapper import load_class

logger = logging.getLogger(__name__)


class DatasetBuilderFactory:
    def __init__(self):
        self.data = {}

    @staticmethod
    def get_class(class_name):
        factory = DatasetBuilderFactory()
        if class_name in factory.data:
            return factory.data[class_name]

        return load_class(class_name)
