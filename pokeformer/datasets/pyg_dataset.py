import copy
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from pokeformer.datasets.base_dataset import BaseDataset
from pokeformer.receptor import Receptor
from pokeformer.utils.pickle_util import (conv_path, read_data, read_list_iter,
                                          write_data)
from pokeformer.vocabs.prot_atom_vocab import ProtAtomVocab
from torch_geometric.data import Data as PygData
from torch_geometric.loader.dataloader import Collater

logger = logging.getLogger(__name__)


@dataclass
class PygDatasetConfig:
    data_pkl: Optional[str] = None
    data_pkl_min_idx: Optional[int] = None
    data_pkl_max_idx: Optional[int] = None
    dataset_frac: Optional[float] = None

    atom_vocab: Optional[str] = None
    prot_atom_vocab: Optional[str] = None

    scratch: Optional[str] = None
    # emb_scratch: Optional[str] = None
    make_scratch: bool = False

    add_receptor: bool = False

    limit_atoms: Optional[List[str]] = None
    limit_data: Optional[int] = None

    use_pkl_stream: bool = True
    remove_unk: bool = False

    iou_threshold: Optional[float] = None
    mindist_threshold: Optional[float] = None

    drop_node_rate: Optional[float] = None
    pos_noise_scale: Optional[float] = None
    drop_node_choice: bool = True
    sasa_noise_scale: Optional[float] = None
    sasa_mul_noise: bool = False

    gpu: int = -1

    # SASA
    use_sasa: Optional[str] = None


class PygDataset(BaseDataset):
    @classmethod
    def get_config_class(cls):
        return PygDatasetConfig

    @classmethod
    def from_config(cls, config):
        prot_vocab = ProtAtomVocab.from_json_file(config.prot_atom_vocab)
        return cls(
            prot_vocab=prot_vocab,
            config=config,
        )

    def __init__(
        self,
        prot_vocab,
        config,
    ):
        # self.device = torch.device(f"cuda:{config.gpu}" if config.gpu >= 0 else "cpu")

        self.aug_enabled = False
        self.drop_node_rate = config.drop_node_rate
        self.drop_node_choice = config.drop_node_choice
        logger.info(f"Use np.random.choice for drop node: {self.drop_node_choice}")
        self.pos_noise_scale = config.pos_noise_scale
        logger.info(f"{self.pos_noise_scale=}")
        self.sasa_noise_scale = config.sasa_noise_scale
        logger.info(f"{self.sasa_noise_scale=}")
        self.sasa_mul_noise = config.sasa_mul_noise
        logger.info(f"{self.sasa_mul_noise=}")

        self.iou_threshold = config.iou_threshold
        self.mindist_threshold = config.mindist_threshold

        self.config = config
        self.limit_atoms = None
        if self.config.limit_atoms is not None:
            lst = [str(i) for i in self.config.limit_atoms]
            logger.info(f"limit atom types to: {lst}")
            self.limit_atoms = lst

        self.prot_vocab = prot_vocab
        self.collater = Collater(follow_batch=None, exclude_keys=None)

        scr_path = None
        if config.scratch is not None:
            scr_path = conv_path(config.scratch)
            if config.make_scratch:
                logger.info(f"forced to create scratch: {config.scratch}")
            elif scr_path.exists():
                self.load_scratch()
                self.data = self.data[: config.limit_data]
                return
            else:
                logger.info(
                    f"scartch {config.scratch} not found --> build from dataset"
                )

        #####

        # Build scratch data
        logger.info("building dataset")
        if self.config.data_pkl is not None:
            self.data = self._pickle_loader(self.config.data_pkl, data_id=0)
        else:
            logger.warning("no data loaded")
            self.data = []

        if scr_path is not None:
            self.write_scratch()

        self.data = self.data[: config.limit_data]

    def write_scratch(self):
        write_data(self.config.scratch, self.data)
        logger.info(f"wrote scratch file: {self.config.scratch}")

    def load_scratch(self):
        logger.info(f"loading from saved scratch: {self.config.scratch}")
        self.data = read_data(self.config.scratch)

        logger.info("DONE")

    def get_metrics_keys(self):
        return []

    def get_vocab_dict(self):
        return {
            "prot_vocab": self.prot_vocab,
        }

    def __len__(self):
        result = len(self.data)
        logger.info(f"len: {result}")
        return result

    def __getitem__(self, item):
        result = self.data[item]
        result.aug = False
        if self.aug_enabled:
            res2 = self.augmentation(result)
            return res2
        return result

    def enable_aug(self, flag):
        self.aug_enabled = flag

    def augmentation(self, dat):
        if self.drop_node_rate is None and self.pos_noise_scale is None:
            return dat

        newdat = copy.copy(dat)
        if self.drop_node_rate is not None:
            num_nodes = len(newdat.x)

            if self.drop_node_choice:
                num_drop = int(num_nodes * self.drop_node_rate)
                ids = np.random.choice(num_nodes, num_nodes - num_drop)
            else:
                ids = np.random.rand(num_nodes) > self.drop_node_rate

            newdat.x = newdat.x[ids]
            newdat.data_id = newdat.data_id[ids]
            newdat.pos = newdat.pos[ids]
            if "sasa" in newdat:
                newdat.sasa = newdat.sasa[ids]
            newdat.aug = True
        if self.pos_noise_scale is not None:
            pos_noise = torch.randn(size=newdat.pos.shape)
            newdat.pos = newdat.pos + pos_noise * self.pos_noise_scale
            newdat.aug = True
            if "sasa" in newdat and self.sasa_noise_scale is not None:
                sasa_noise = torch.randn(size=newdat.sasa.shape) * self.sasa_noise_scale
                if not self.sasa_mul_noise:
                    # Additive noise
                    newdat.sasa = torch.nn.functional.relu(newdat.sasa + sasa_noise)
                else:
                    # Multiplicative noise
                    newdat.sasa = newdat.sasa * torch.nn.functional.relu(1 + sasa_noise)
        return newdat

    def assign_label(self, prot_data):
        if self.iou_threshold is not None:
            assert self.mindist_threshold is None
            iou = prot_data["pocket_info"]["iou"]
            return iou > self.iou_threshold
        elif self.mindist_threshold is not None:
            mindist = prot_data["pocket_info"]["dist_min"]
            return mindist < self.mindist_threshold
        else:
            raise RuntimeError("both iou and mindist are null")

    def getitem_impl(self, prot_data):
        # Label (0 or 1)
        if "label" not in prot_data:
            prot_data["label"] = self.assign_label(prot_data)
        tgt_label = prot_data["label"]
        tgt_label = torch.tensor([tgt_label], dtype=torch.float)

        prot = Receptor(prot_data["prot_atom_feat"], prot_data["PDB_ID"])
        if self.limit_atoms is not None:
            prot = prot.filter_atoms(lambda d: d["atom_name"] not in self.limit_atoms)

        # Protein nodes (atoms)
        prot_atoms = self.prot_vocab.to_seq(prot, with_eos=False, with_sos=False)
        prot_atoms = torch.tensor(prot_atoms, dtype=torch.long)

        # Protein nodes (coords)
        prot_crds = prot.get_np_coord()
        prot_crds = torch.tensor(prot_crds)

        sasa = None
        if self.config.use_sasa is not None:
            try:
                sasa = [atm["sasa_resid"] for atm in prot.data]
                sasa = torch.tensor(sasa, dtype=torch.float)
            except:
                pass

        # remove nodes with the unknown label
        if self.prot_vocab.unk_index in prot_atoms:
            if self.config.remove_unk:
                logger.warning(
                    f"unkown token in {prot_data['PDB_ID']=} {prot_atoms=} removed"
                )
                mask = prot_atoms != self.prot_vocab.unk_index
                prot_atoms = prot_atoms[mask]
                prot_crds = prot_crds[mask]
            else:
                raise RuntimeError(
                    f"unkown token in {prot_data['PDB_ID']=} {prot_atoms=}"
                )

        if len(prot_atoms) == 0:
            logger.error(f"{prot=}")
            raise AssertionError()

        if "data_id" in prot_data:
            data_id = prot_data["data_id"]
        else:
            data_id = 0
        data = PygData(
            x=prot_atoms,
            y=tgt_label,
            pos=prot_crds,
            data_id=torch.full_like(prot_atoms, fill_value=data_id),
        )

        if "Score" in prot_data["pocket_info"]:
            data["fpocket_score"] = prot_data["pocket_info"]["Score"]

        if "fold_id" in prot_data:
            data["fold_id"] = prot_data["fold_id"]

        if sasa is not None:
            data["sasa"] = sasa

        if self.config.add_receptor:
            data["PDB_ID"] = prot_data["PDB_ID"]

        return data

    def get_fold_id(self, ind):
        return self.data[ind].fold_id

    def collate_fn(self, data_list):
        result = self.collater(data_list)
        return result

    def _pickle_loader(self, data_pkl, data_id=0):
        config = self.config

        min_idx = 0
        max_idx = sys.maxsize
        if config.data_pkl_min_idx is not None:
            min_idx = config.data_pkl_min_idx
        if config.data_pkl_max_idx is not None:
            max_idx = config.data_pkl_max_idx
        data = []
        for i in range(min_idx, max_idx):
            d = self._pickle_load_one_file(data_pkl, i, data_id)
            if d is None:
                break
            data.extend(d)
        if len(data) == 0:
            raise RuntimeError(f"no mols read from {config.data_pkl}")

        # Apply overall reduction by fraction
        assert data is not None
        nsize = len(data)
        if config.dataset_frac is not None:
            nsize = int(len(data) * config.dataset_frac)
            data = data[:nsize]
        logger.info(f"loaded size: {nsize}")
        return data

    def _pickle_load_one_file(self, data_pkl, ind, data_id=0):
        if ind is None:
            fn = data_pkl
        else:
            fn = data_pkl.format(ind)
        fn_path = conv_path(fn)
        if not fn_path.exists():
            logger.warning(f"input pkl file not found: {fn}")
            return None
        with fn_path.open("rb") as f:
            logger.info(f"loading file: {fn}")
            assert self.config.use_pkl_stream, "only pickle stream format supported"

            orig_dat_iter = read_list_iter(f)
            cur_dat = []
            for dat in orig_dat_iter:
                if dat is None:
                    logger.warning("data is none")
                    continue
                dat["data_id"] = data_id
                pyg_data = self.getitem_impl(dat)
                if pyg_data is None:
                    logger.warning("PyG data is none")
                    continue
                cur_dat.append(pyg_data)

        return cur_dat
