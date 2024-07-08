import logging
from collections import defaultdict

from Bio.PDB import Selection
from pokeformer.utils.pdb_utils import calc_id_model, get_biopdb_struct

import freesasa

logger = logging.getLogger(__name__)


coeff = 0.9033294541493975
intcep = 5.528451433776809


def calc_sasa(pdb_path, name="protein"):
    struct = get_biopdb_struct(name, pdb_path)
    result, sasa_classes = freesasa.calcBioPDB(struct)

    ra = result.residueAreas()
    sasa_resid = defaultdict(float)
    resids = Selection.unfold_entities(struct, "R")
    for i, res in enumerate(resids):
        full_id = res.get_full_id()
        idx = calc_id_model(res)
        chain = full_id[2]
        res_idx = str(full_id[3][1])
        if chain in ra and res_idx in ra[chain]:
            area = ra[chain][res_idx]
            sasa_resid[idx] = area.total * coeff + intcep

    return sasa_resid
