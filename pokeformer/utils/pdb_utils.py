import gzip
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB import (PDBIO, NeighborSearch, PDBParser, Polypeptide, Select,
                     Selection)
from pokeformer.consts import PDB_SET_DIR

logger = logging.getLogger(__name__)


_hydrogen = re.compile("[123 ]*[HD].*")


def _open_gz(path):
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt")
    else:
        return path.open("r")


def get_biopdb_struct(name, pdb_path):
    with _open_gz(pdb_path) as f:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(name, f)
        return struct


def get_pdb_path(pdbid, pdb_set_dir=None, upper=True):
    if upper:
        subdir = f"{pdbid[1:3].upper()}/{pdbid.upper()}.pdb1.gz"
    else:
        subdir = f"{pdbid[1:3].lower()}/{pdbid.lower()}.pdb1.gz"
    if pdb_set_dir is None:
        return PDB_SET_DIR / subdir
    else:
        return Path(pdb_set_dir) / subdir


def is_hydrogen(atom):
    return _hydrogen.match(atom.get_id())


class StdAASelect(Select):
    def __init__(self, data):
        self.data = data

    def accept_residue(self, residue):
        # remove HETATMs
        if residue.get_full_id()[3][0] != " ":
            return False
        return Polypeptide.is_aa(residue, standard=True)

    def accept_atom(self, atom):
        # remove Hs
        if is_hydrogen(atom):
            return False

        if atom.get_full_id() not in self.data:
            return False

        # remove altconfs
        altloc = self.data[atom.get_full_id()]
        if altloc != atom.get_altloc():
            return False

        return True


class BasicStdAASelect(Select):
    def accept_residue(self, residue):
        return Polypeptide.is_aa(residue, standard=True)


def write_clean_pdb(struct, f, remove_alt_h=False):
    if remove_alt_h:
        # check altconf
        atoms = Selection.unfold_entities(struct, "A")
        altconfs = {}
        for i, a in enumerate(atoms):
            a_full_id = a.get_full_id()
            altconfs[a_full_id] = a.get_altloc()
        selector = StdAASelect(altconfs)
    else:
        selector = BasicStdAASelect()

    io = PDBIO()
    io.set_structure(struct)
    io.save(f, selector)


def read_pqr(pqr_path, radii=True):
    coord = []
    with open(pqr_path, mode="r") as f:
        for ln in f:
            if ln.startswith("ATOM  "):
                atomline = ln
                if radii:
                    xyzr = [
                        float(atomline[30:38]),  # x coord
                        float(atomline[38:46]),  # y coord
                        float(atomline[46:54]),  # z coord
                        float(atomline[62:71]),  # radius
                    ]
                    coord.append(xyzr)
                else:
                    xyz = [
                        float(atomline[30:38]),  # x coord
                        float(atomline[38:46]),  # y coord
                        float(atomline[46:54]),  # z coord
                    ]
                    coord.append(xyz)
    return np.asarray(coord)


def get_neighbor_atoms(
    struct, pkt_verts, dist=5.0, byresidue=True, remove_hs=True, name="protein"
):
    atoms = Selection.unfold_entities(struct, "A")
    ns = NeighborSearch(atoms)

    prot_catms = []
    lig_catms = []
    for v in pkt_verts:
        if byresidue:
            # Residue-level search
            result = ns.search(center=v, radius=dist, level="R")
            # Unfold to atoms from residues
            cat = Selection.unfold_entities(result, "A")
        else:
            cat = ns.search(center=v, radius=dist, level="A")
        for atm in cat:
            resid = atm.get_parent()
            if Polypeptide.is_aa(resid, standard=True):
                prot_catms.append(atm)
            else:
                lig_catms.append(atm)

    # uniquefy
    prot_catms = sorted(list(set(prot_catms)))
    lig_catms = sorted(list(set(lig_catms)))

    if remove_hs:
        prot_catms = [atm for atm in prot_catms if not is_hydrogen(atm)]

    return prot_catms, lig_catms


struct_cache = {}


def get_nei_atoms_by_pdbid(
    pdbid, pkt_verts, dist=5.0, byresidue=True, remove_hs=True, pdb_set_dir=None
):
    if pdbid in struct_cache:
        clean_struct = struct_cache[pdbid]
    else:
        pdb_path = get_pdb_path(pdbid, pdb_set_dir=pdb_set_dir)
        with _open_gz(pdb_path) as f:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure(pdbid, f)
        with tempfile.TemporaryDirectory() as dir_name:
            dir_path = Path(dir_name)
            tmp_pdb_path = dir_path / "in.pdb"
            with tmp_pdb_path.open("w") as f:
                write_clean_pdb(struct, f)

            with tmp_pdb_path.open("r") as f:
                parser = PDBParser(QUIET=True)
                clean_struct = parser.get_structure(pdbid, f)
        struct_cache[pdbid] = clean_struct

    return get_neighbor_atoms(
        clean_struct,
        pkt_verts,
        dist=dist,
        byresidue=byresidue,
        remove_hs=remove_hs,
        name=pdbid,
    )


def get_nei_atoms_by_pdbfile(
    pdb_path, pkt_verts, dist=5.0, byresidue=True, remove_hs=True
):
    with _open_gz(pdb_path) as f:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(str(pdb_path), f)

    with tempfile.TemporaryDirectory() as dir_name:
        dir_path = Path(dir_name)
        tmp_pdb_path = dir_path / "in.pdb"
        with tmp_pdb_path.open("w") as f:
            write_clean_pdb(struct, f)

        with tmp_pdb_path.open("r") as f:
            parser = PDBParser(QUIET=True)
            clean_struct = parser.get_structure(str(pdb_path), f)

    return get_neighbor_atoms(
        clean_struct,
        pkt_verts,
        dist=dist,
        byresidue=byresidue,
        remove_hs=remove_hs,
        name=pdb_path,
    )


def calc_id_model(resid):
    # FID: (name, model_id, chain_name, (hetero, seqid, inscode))
    fid = resid.get_full_id()
    name = fid[0]
    model_id = fid[1]
    chain_name = fid[2]
    res_fid = fid[3]
    resn = resid.get_resname()
    seqid = res_fid[1]
    insc = res_fid[2]
    return (name, model_id, chain_name, resn, seqid, insc)


def calc_id(resid):
    # FID: (name, model_id, chain_name, (hetero, seqid, inscode))
    fid = resid.get_full_id()
    name = fid[0]
    chain_name = fid[2]
    res_fid = fid[3]
    resn = resid.get_resname()
    seqid = res_fid[1]
    insc = res_fid[2]
    return (name, chain_name, resn, seqid, insc)


def get_com_coords(coords):
    com = coords.mean(axis=0)
    return com


def calc_min_dist(crds1, crds2):
    diff = crds1[None, :, :] - crds2[:, None, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = np.min(dist)
    return min_dist


##########


def get_ligand_coords(pdb_path, het_code=None):
    coord = []
    with _open_gz(pdb_path) as f:
        for ln in f:
            if ln.startswith("HETATM"):
                # if ln.startswith("ATOM  "):
                atomline = ln
                xyz = [
                    float(atomline[30:38]),  # x coord
                    float(atomline[38:46]),  # y coord
                    float(atomline[46:54]),  # z coord
                ]
                if het_code is None:
                    coord.append(xyz)
                    continue
                # check residue name
                resn = atomline[17:20].strip()
                if het_code == resn:
                    coord.append(xyz)

    coords = np.asarray(coord)
    return coords


def check_covalent_ligand(pdb_path, coval_thr):
    lig_coord = []
    pro_coord = []
    with _open_gz(pdb_path) as f:
        for ln in f:
            if ln.startswith("HETATM"):
                # if ln.startswith("ATOM  "):
                atomline = ln
                xyz = [
                    float(atomline[30:38]),  # x coord
                    float(atomline[38:46]),  # y coord
                    float(atomline[46:54]),  # z coord
                ]
                lig_coord.append(xyz)
            elif ln.startswith("ATOM  "):
                atomline = ln
                xyz = [
                    float(atomline[30:38]),  # x coord
                    float(atomline[38:46]),  # y coord
                    float(atomline[46:54]),  # z coord
                ]
                pro_coord.append(xyz)

    lig_coords = np.asarray(lig_coord)
    pro_coords = np.asarray(pro_coord)
    diff = lig_coords[None, :, :] - pro_coords[:, None, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = np.min(dist)

    if min_dist < coval_thr:  # COVAL_THR:
        return True
    else:
        return False
