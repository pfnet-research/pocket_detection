import logging
from pathlib import Path

from pokeformer.preproc.freesasa import calc_sasa
from pokeformer.utils.pdb_utils import (calc_id, calc_id_model,
                                        get_nei_atoms_by_pdbfile,
                                        get_nei_atoms_by_pdbid, get_pdb_path)

logger = logging.getLogger(__name__)


def fmt_prot_data1(pkt_prot_atoms):
    feat_ent = []
    for a in pkt_prot_atoms:
        resid = a.get_parent()
        fid = resid.get_full_id()
        chname = fid[2]
        res_fid = fid[3]
        res_id = res_fid[1]
        resn = resid.get_resname()
        xyz = a.get_coord()
        dat = {
            "asm_id": calc_id_model(resid),
            "seq_id": calc_id(resid),
            "atom_name": a.get_name(),
            "res_name": resn,
            "ch_name": chname,
            "res_id": res_id,
            "coord": xyz,
        }
        feat_ent.append(dat)
    return feat_ent


def add_sasa_data(feat_ent, sasa_resid):
    for ent in feat_ent:
        asm_id = ent["asm_id"]
        assert asm_id in sasa_resid, f"{asm_id=} not found in sasa_resid"
        ent["sasa_resid"] = sasa_resid[asm_id]
    return feat_ent


sasa_cache = {}


def get_prot_feats_around_pocket(dat, config):
    around_dist = config.pocket_around_dist
    pkt_verts = dat["pocket_verts"]

    # remove radius data (if exists)
    pkt_verts = pkt_verts[:, 0:3]

    pdbid = dat["PDB_ID"]
    if len(pdbid) == 4:
        pdb_path = get_pdb_path(pdbid, config.pdb_set_dir)
        logger.debug(f"{config.pdb_set_dir=}")
        logger.debug(f"{pdb_path=}")
        pkt_prot_atoms, _ = get_nei_atoms_by_pdbid(
            pdbid, pkt_verts=pkt_verts, dist=around_dist, pdb_set_dir=config.pdb_set_dir
        )
    else:
        # PDB_ID is actually pdb_path
        pdb_path = Path(pdbid)
        pkt_prot_atoms, _ = get_nei_atoms_by_pdbfile(
            pdb_path, pkt_verts=pkt_verts, dist=around_dist
        )

    result = fmt_prot_data1(pkt_prot_atoms)
    if config.calc_sasa is not None:
        assert pdb_path.exists(), f"not found: {pdb_path}"
        if pdbid in sasa_cache:
            sasa = sasa_cache[pdbid]
        else:
            try:
                sasa = calc_sasa(pdb_path, pdbid)
            except Exception:
                logger.warning(f"preproc failed for {pdbid=}", exc_info=True)
                sasa = None
            sasa_cache[pdbid] = sasa
        if sasa is None:
            raise RuntimeError(f"sasa calc failed for {pdbid=}")
        result = add_sasa_data(result, sasa)

    return result
