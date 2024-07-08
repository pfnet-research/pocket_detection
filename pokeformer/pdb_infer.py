import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import tqdm
from omegaconf import MISSING
from torch.utils.data import DataLoader

from pokeformer.preproc.detect_pocket import detect_pocket, detect_pocket_file
from pokeformer.preproc.ligand import (calc_ligand_pocket_distance,
                                       get_ligand_data)
from pokeformer.preproc.prot_feature import get_prot_feats_around_pocket

logger = logging.getLogger(__name__)


@dataclass
class PDBInferConfig:
    # Input
    pdbid_csv: Optional[str] = None
    pdb_files: List[str] = field(default_factory=list)
    pdbids: List[str] = field(default_factory=list)
    pdb_set_dir: Optional[str] = None
    ksite_csv: Optional[str] = None

    start_idx: int = 0
    end_idx: int = -1

    # Output
    out_csv: str = MISSING

    # SASA
    calc_sasa: bool = True

    # Model
    gpu: int = 0
    model_path_list: List[str] = MISSING
    batch_size: int = 32

    backend: str = "fpocket"
    pocket_bin_dir: Optional[str] = None
    reuse_fpocket_result: bool = False

    pocket_result_base: Optional[str] = None

    rs: float = 1.87
    rlx: float = 3.0
    br: float = 1.0

    # pocket criteria
    grid_size: float = 0.5
    min_vol: float = 0.0
    max_vol: float = 1e10
    pocket_around_dist: float = 10.0

    lig_radius: float = 1.6


OUTPUT_KEYS = [
    "PDB_ID",
    "dist_min",
    "dist_min_het_code",
    "volume",
    "label",
    "HET_Code",
]


class PDBInfer:
    @classmethod
    def get_config_class(cls):
        return PDBInferConfig

    @classmethod
    def from_config(cls, config, *args):
        return cls(config, *args)

    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu}" if config.gpu >= 0 else "cpu")

        if self.config.ksite_csv is not None:
            self.ksite_df = pd.read_csv(self.config.ksite_csv, index_col=0)
        else:
            self.ksite_df = None

    def infer_batch(self, batch, model):
        batch = batch.to(self.device)
        results = model(batch)
        label = batch["y"].cpu().detach().numpy()
        preds = results["preds"].cpu().detach().numpy()
        logger.debug(f"{preds=}")
        logger.debug(f"{label=}")
        return preds, label

    def run(self, model, dataset):
        results = None
        pred_keys = []
        loader, data = self.create_dataset(dataset)
        for i, model_path in enumerate(self.config.model_path_list):
            if i == 0:
                add_aux_info = True
            else:
                add_aux_info = False
            pred_key = f"pred_{i}"
            pred_keys.append(pred_key)

            self.load_model(model, model_path)
            res = self.run_one_model(
                model, loader, data, pred_key=pred_key, add_aux_info=add_aux_info
            )
            res = pd.DataFrame(res)
            if results is None:
                results = res
            else:
                results[pred_key] = res[pred_key]

        results["pred_aver"] = results[pred_keys].mean(axis=1)
        results["pred_std"] = results[pred_keys].std(axis=1)
        results.to_csv(self.config.out_csv)

    def create_dataset(self, dataset):
        config = self.config
        if config.pdbid_csv is not None:
            pdb_df = pd.read_csv(config.pdbid_csv, index_col=0)
            pdbids = pdb_df["PDB_ID"].unique().tolist()
            test_set, orig_data = self.create_test_dataset(dataset, pdbids)
        elif len(config.pdbids) > 0:
            pdbids = [str(i) for i in config.pdbids]
            test_set, orig_data = self.create_test_dataset(dataset, pdbids)
        elif len(config.pdb_files) > 0:
            pdbfiles = [str(i) for i in config.pdb_files]
            test_set, orig_data = self.create_data_by_pdbfiles(dataset, pdbfiles)
        else:
            raise RuntimeError("invalid pdbid input config")

        loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=test_set.collate_fn,
        )
        return loader, orig_data

    def run_one_model(
        self, model, loader, orig_data, pred_key="pred", add_aux_info=True
    ):
        config = self.config
        res_dat = []
        for batch in tqdm.tqdm(loader, total=(len(orig_data) // config.batch_size) + 1):
            with torch.no_grad():
                preds, labels = self.infer_batch(batch, model)
                for i in range(len(batch)):
                    res = {}
                    if add_aux_info:
                        idx = int(batch[i].idx)
                        res["idx"] = idx
                        for key in OUTPUT_KEYS:
                            if key in orig_data[idx]:
                                res[key] = orig_data[idx][key]
                    res[pred_key] = preds[i]
                    logger.info(f"{res=}")
                    res_dat.append(res)

        return res_dat

    def create_data_by_pdbid(self, pdbid):
        pocket_path = None
        if self.config.pocket_result_base is not None:
            pocket_path = Path(self.config.pocket_result_base) / pdbid

        try:
            pkt_strs, pkt_info_set = detect_pocket(
                pdbid, self.config, save_result_path=pocket_path
            )
        except RuntimeError:
            return None

        if self.ksite_df is not None:
            lig_list = get_ligand_data(pdbid, self.ksite_df, self.config.pdb_set_dir)
            calc_ligand_pocket_distance(lig_list, pkt_strs, pkt_info_set)

        results = self._convert_results(
            pdbid=pdbid, pkt_strs=pkt_strs, pkt_info_set=pkt_info_set
        )

        logger.info(f"{pdbid=} {len(results)=}")
        if len(results) == 0:
            logger.warning(f"{pdbid=} no pocket found")
            return None

        return results

    def create_test_dataset(self, dataset, pdbids):
        dat_list = []

        pdbids = pdbids[self.config.start_idx : self.config.end_idx]

        for pdbid in tqdm.tqdm(pdbids):
            d = self.create_data_by_pdbid(pdbid)
            if d is not None:
                dat_list.extend(d)
        logger.info(f"dat_list: {len(dat_list)=}")

        for i in range(len(dat_list)):
            d = dataset.getitem_impl(dat_list[i])
            d["idx"] = i
            dataset.data.append(d)
        return dataset, dat_list

    def create_data_by_pdbfiles(self, dataset, pdb_files):
        params = self.config
        dat_list = []
        for pdb_file in tqdm.tqdm(pdb_files):
            pocket_path = None
            if self.config.pocket_result_base is not None:
                subdir = Path(pdb_file).name
                pocket_path = Path(self.config.pocket_result_base) / subdir
            try:
                pkt_strs, pkt_info_set = detect_pocket_file(
                    pdb_file,
                    params,
                    save_result_path=pocket_path,
                )
            except RuntimeError:
                continue

            results = self._convert_results(
                pdbid=pdb_file, pkt_strs=pkt_strs, pkt_info_set=pkt_info_set
            )

            logger.info(f"{pdb_file=} {len(results)=}")
            if len(results) == 0:
                logger.warning(f"{pdb_file=} no pocket found")
                continue
            dat_list.extend(results)
        logger.info(f"dat_list: {len(dat_list)=}")

        for i in range(len(dat_list)):
            d = dataset.getitem_impl(dat_list[i])
            d["idx"] = i
            dataset.data.append(d)
        return dataset, dat_list

    def _convert_results(self, pdbid, pkt_strs, pkt_info_set):
        max_vol = self.config.max_vol
        min_vol = self.config.min_vol
        results = []
        for pkt_id in pkt_strs.keys():
            pkt_info = pkt_info_set[pkt_id]
            dat = {
                "PDB_ID": pdbid,
                "pocket_info": pkt_info,
                "pocket_verts": pkt_strs[pkt_id],
            }

            vol = dat["volume"] = float(dat["pocket_info"]["Volume"])
            logger.debug(f"{vol=} {type(vol)=}")
            if vol > max_vol or vol < min_vol:
                continue

            if "iou" in dat["pocket_info"]:
                dat["iou"] = dat["pocket_info"]["iou"]
            else:
                dat["iou"] = dat["pocket_info"]["iou"] = 0.0

            if "dist_min" in dat["pocket_info"]:
                dat["dist_min"] = dat["pocket_info"]["dist_min"]
                dat["dist_imin"] = dat["pocket_info"]["dist_imin"]
                dat["dist_min_het_code"] = dat["pocket_info"]["dist_min_het_code"]
            else:
                dat["dist_min"] = dat["pocket_info"]["dist_min"] = 999.0

            dat["prot_atom_feat"] = get_prot_feats_around_pocket(dat, self.config)
            results.append(dat)
        return results

    def load_model(self, model, model_path):
        model.to(self.device)
        snapshot = model_path
        logger.info(f"Load snapshot from {snapshot}")
        state = torch.load(snapshot)
        if "models" in state:
            model.load_state_dict(state["models"]["main"])
        else:
            model.load_state_dict(state)

        # Show model info
        try:
            import torchinfo

            # torchinfo.summary(model, verbose=2)
            torchinfo.summary(model, verbose=1)
        except Exception:
            logger.info("cannot call torchinfo.summary", exc_info=True)
