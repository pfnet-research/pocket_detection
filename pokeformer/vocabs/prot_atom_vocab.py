import logging
from collections import Counter

from pokeformer.vocabs.vocab import Vocab

logger = logging.getLogger(__name__)


class ProtAtomFeature:
    def __init__(self, res_name=None, atom_name=None):
        if res_name is None:
            return
        self.res_name = res_name
        self.atom_name = atom_name

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def to_tuple(self):
        return (self.res_name, self.atom_name)

    def __hash__(self):
        return hash(self.to_tuple())

    def __lt__(self, other):
        # self < other
        return self.to_tuple() < other.to_tuple()

    def __repr__(self):
        return f"{self.res_name}:{self.atom_name}"

    def to_dict(self):
        result = {
            "res_name": self.res_name,
            "atom_name": self.atom_name,
        }
        return result

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.res_name = d["res_name"]
        obj.atom_name = d["atom_name"]
        return obj


class ProtAtomVocab(Vocab):
    def __init__(self, mols=None):
        if mols is None:
            return

        self.atom_feature_cls = ProtAtomFeature
        counter = Counter()
        for mol in mols:
            for atom in mol.get_atom_feats():
                word = atom
                counter[word] = 1
        logger.info(f"vocab size: {len(counter)}")
        specials = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"]
        super().__init__(counter, specials)
        logger.info(f"built vocab: {self.itos}")
        logger.info(f"built vocab: {self.stoi}")

    def to_seq(self, mol, with_eos=False, with_sos=False, warn_unk=True):
        seq = [self.stoi.get(word, self.unk_index) for word in mol.get_atom_feats()]

        if self.unk_index in seq:
            logger.warning(f"Unknown token (self.unk_index) in {seq=}")
            feats = list(mol.get_atom_feats())
            unks = [feats[ix] for ix, v in enumerate(seq) if v == self.unk_index]
            logger.warning(f"Unknown atom names: {unks}")

        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq

        logger.debug(f"seq: {seq}")
        return seq

    def from_seq(self, seq):
        words = []
        for idx in seq:
            if idx == self.pad_index:
                continue
            words.append(self.itos[idx])
        return words

    def to_dict(self):
        result = super().to_dict()
        result["atom_feature"] = "ProtAtomFeature"
        return result

    @classmethod
    def from_dict(cls, d):
        afcls = ProtAtomFeature

        new_itos = [afcls.from_dict(v) for v in d["itos"]]
        d["itos"] = new_itos

        obj = super().from_dict(d)
        obj.atom_feature_cls = afcls
        return obj
