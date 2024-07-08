import json
import logging

logger = logging.getLogger(__name__)


def _to_dict(value):
    if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
        return value
    else:
        return value.to_dict()


class BaseVocab(object):
    def __init__(
        self,
        counter=None,
        specials=None,
    ):
        if counter is None:
            return
        if specials is None:
            specials = list()

        self.ctl_words = [i for i in specials]
        for tok in specials:
            del counter[tok]

        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])

        self.itos = [i for i in specials]
        for word, _ in words_and_frequencies:
            self.itos.append(word)

        self.build_stoi()

    def build_itos(self, ctl_words, words):
        self.ctl_words = [i for i in ctl_words]
        self.itos = [i for i in ctl_words] + sorted(words)

    def build_stoi(self):
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def is_ctl_word(self, w):
        return isinstance(w, str) and w in self.ctl_words

    def merge(self, voc):
        assert self.ctl_words == voc.ctl_words
        # build i-->s mapping
        new_itos = [i for i in self.itos if not self.is_ctl_word(i)]
        for i in voc.itos:
            if not self.is_ctl_word(i) and i not in new_itos:
                new_itos.append(i)
        self.build_itos(self.ctl_words, new_itos)

        # rebuild s-->i mapping
        self.build_stoi()

    def __len__(self):
        return len(self.itos)

    def __eq__(self, value):
        stoi1 = sorted(self.stoi.items(), key=lambda x: x[1])
        stoi2 = sorted(value.stoi.items(), key=lambda x: x[1])
        v2 = [k1 == k2 and v1 == v2 for (k1, v1), (k2, v2) in zip(stoi1, stoi2)]
        v1 = [v1 == v2 for v1, v2 in zip(self.itos, value.itos)]
        logger.debug(f"__eq__: {v1}, {v2}")
        return all(v1 + v2)

    @property
    def pad_index(self):
        return self.stoi["<pad>"]

    @property
    def unk_index(self):
        return self.stoi["<unk>"]

    @property
    def eos_index(self):
        return self.stoi["<eos>"]

    @property
    def sos_index(self):
        return self.stoi["<sos>"]

    @property
    def mask_index(self):
        return self.stoi["<mask>"]

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        raise NotImplementedError()

    def from_seq(self, seq, join=False, with_pad=False):
        raise NotImplementedError()

    def to_dict(self):
        # stoi_dict = [(_to_dict(k), v) for k, v in self.stoi.items()]
        itos_dict = [_to_dict(v) for v in self.itos if not self.is_ctl_word(v)]
        # result = {"stoi": stoi_dict, "itos": itos_dict}
        result = {"ctl_words": self.ctl_words, "itos": itos_dict}
        return result

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_json_file(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.build_itos(d["ctl_words"], d["itos"])
        obj.build_stoi()
        return obj

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, file_name):
        with open(file_name, "r") as f:
            return cls.from_dict(json.load(f))
