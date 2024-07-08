import pickle

from .base_vocab import BaseVocab


class Vocab(BaseVocab):
    @staticmethod
    def load_vocab(vocab_path: str) -> "Vocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
