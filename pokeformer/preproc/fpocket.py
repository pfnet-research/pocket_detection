import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pokeformer.utils.pdb_utils import (get_biopdb_struct, read_pqr,
                                        write_clean_pdb)

logger = logging.getLogger(__name__)


def run_fpocket(
    pdb_path,
    print_info=False,
    load_radii=False,
    save_result_path=None,
    remove_alt_h=False,
    bin_dir=None,
    reuse_result=False,
):
    struct = get_biopdb_struct("protein", Path(pdb_path))

    if save_result_path is None:
        with tempfile.TemporaryDirectory() as dir_name:
            dir_path = Path(dir_name)
            return _run_fpocket_core(
                struct,
                dir_path,
                print_info,
                load_radii,
                remove_alt_h=remove_alt_h,
                bin_dir=bin_dir,
            )
    else:
        dir_path = Path(save_result_path)
        if reuse_result:
            pkts, info_dat = _parse_out_data(dir_path, print_info, load_radii)
            return pkts, info_dat
        else:
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
            dir_path.mkdir(exist_ok=True, parents=True)
            return _run_fpocket_core(
                struct,
                dir_path,
                print_info,
                load_radii,
                remove_alt_h=remove_alt_h,
                bin_dir=bin_dir,
            )


def _run_fpocket_core(
    struct,
    dir_path,
    print_info=False,
    load_radii=False,
    remove_alt_h=False,
    bin_dir=None,
):
    in_pdb_path = dir_path / "in.pdb"
    with in_pdb_path.open("w") as f:
        write_clean_pdb(struct, f, remove_alt_h=remove_alt_h)

    if bin_dir is None:
        fpocket_bin = "fpocket"
    else:
        fpocket_bin = str(Path(bin_dir) / "fpocket")

    cmd = [fpocket_bin, "-f", str(in_pdb_path)]
    subprocess.run(cmd)

    try:
        pkts, info_dat = _parse_out_data(dir_path, print_info, load_radii)
    except Exception:
        logger.error(f"fpocket run failed for {in_pdb_path}")
        raise

    return pkts, info_dat


def _parse_out_data(dir_path, print_info=False, load_radii=False):
    out_dir = dir_path / "in_out" / "pockets"
    result_file = dir_path / "in_out" / "in_info.txt"
    with result_file.open("r") as f:
        data = f.read()

    # parse info file --> k-v data
    pkt_id = None
    info_dat = {}
    for ln in data.splitlines():
        x = re.match(r"Pocket (\d+) :", ln)
        if x is not None:
            pkt_id = int(x.group(1))
            info_dat[pkt_id] = {}
            continue

        x = re.match(r"\s+(\S.+\S)\s*:\s+([\d\.\-\+]+)", ln)
        if x is not None:
            assert pkt_id is not None
            key = x.group(1)
            value = x.group(2)
            info_dat[pkt_id][key] = value
            continue

    if print_info:
        logger.info(data)
        for f in (dir_path / "in_out").glob("*"):
            logger.info(f"{f=}")
        for f in out_dir.glob("*"):
            logger.info(f"{f=}")

    pkts = {}
    for pkt_id in info_dat.keys():
        pkt_pqr = out_dir / f"pocket{pkt_id}_vert.pqr"
        pkt_crds = read_pqr(pkt_pqr, radii=load_radii)
        pkts[pkt_id] = pkt_crds

    logger.info(f"{list(pkts.keys())=}")
    logger.info(f"{list(info_dat.keys())=}")
    return pkts, info_dat
