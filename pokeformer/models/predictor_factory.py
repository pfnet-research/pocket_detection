import logging

from pokeformer.config_wrapper import load_class

logger = logging.getLogger(__name__)


class PredictorFactory:
    def __init__(self):
        self.data = {}

    def create(self, config, vocab_dict):
        cls = self.data[config.model]
        result = cls(config, vocab_dict)
        return result

    @staticmethod
    def get_class(class_name):
        factory = PredictorFactory()
        if class_name in factory.data:
            return factory.data[class_name]
        return load_class(class_name)
