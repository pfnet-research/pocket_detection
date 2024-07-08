import gzip
import logging

import numpy as np
from Bio.PDB import PDBParser
from pokeformer.utils.pdb_utils import (calc_min_dist, get_com_coords,
                                        get_pdb_path)

logger = logging.getLogger(__name__)


def get_ligand_data(pdbid, ksite_df, pdb_set_dir=None):
    pdbid = pdbid.upper()
    df_tgt = ksite_df[ksite_df["PDB_ID"] == pdbid]

    lig_list = []
    for row in df_tgt.to_dict("records"):
        lig_list.append(row["HET_Code"])
    logger.info(f"{lig_list=}")

    pdb_path = get_pdb_path(pdbid, pdb_set_dir)
    parser = PDBParser(QUIET=True)
    with gzip.open(pdb_path, mode="rt") as f:
        struct = parser.get_structure("protein", f)

    ligands = []
    for model in struct:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                if res_name not in lig_list:
                    continue

                coords = [atom.get_coord() for atom in residue]
                coords = np.stack(coords)

                ligands.append({"HET_Code": res_name, "lig_crds": coords})

    logger.info(f"{ligands=}")

    return ligands


def calc_ligand_pocket_distance(
    lig_list,
    pkt_strs,
    pkt_info_set,
):
    for k, pkt_crds in pkt_strs.items():
        logger.info(f"processing pocket {k=} {len(pkt_crds)=}...")
        pkt_com = get_com_coords(pkt_crds)[None, :3]
        dist_list = [calc_min_dist(ii["lig_crds"], pkt_com) for ii in lig_list]
        dist_list = np.asarray(dist_list)
        imin = np.argmin(dist_list)
        min_het_code = lig_list[imin]["HET_Code"]

        logger.info(f"{dist_list[imin]=}")

        pkt_info_set[k]["dist_min"] = dist_list[imin]
        pkt_info_set[k]["dist_imin"] = imin
        pkt_info_set[k]["dist_min_het_code"] = min_het_code
