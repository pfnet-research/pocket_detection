from __future__ import annotations

import logging

from pokeformer.preproc.fpocket import run_fpocket
from pokeformer.utils.pdb_utils import get_pdb_path

logger = logging.getLogger(__name__)


def _detect_pocket_impl(
    pdb_path,
    params,
    save_result_path=None,
    remove_alt_h=False,
):
    if params.backend == "fpocket":
        pkt_strs, pkt_info_set = run_fpocket(
            pdb_path,
            print_info=False,
            load_radii=True,
            save_result_path=save_result_path,
            remove_alt_h=remove_alt_h,
            bin_dir=params.pocket_bin_dir,
            reuse_result=params.reuse_fpocket_result,
        )
    else:
        raise RuntimeError(f"unknown backend: {params.backend}")

    return pkt_strs, pkt_info_set


def detect_pocket(
    pdbid,
    params,
    save_result_path=None,
    remove_alt_h=False,
):
    logger.info("--------------------")
    logger.info(f"{pdbid=}")
    pdb_path = get_pdb_path(pdbid, params.pdb_set_dir)
    return _detect_pocket_impl(pdb_path, params, save_result_path, remove_alt_h)


def detect_pocket_file(pdb_file, params, save_result_path=None, remove_alt_h=False):
    logger.info("--------------------")
    return _detect_pocket_impl(pdb_file, params, save_result_path, remove_alt_h)
