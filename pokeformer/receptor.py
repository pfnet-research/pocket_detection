import numpy as np

from pokeformer.vocabs.prot_atom_names import atom_names, res_names
from pokeformer.vocabs.prot_atom_vocab import ProtAtomFeature


def _get_pdb_atom_lines(pdb_file: str):
    with open(pdb_file) as f:
        raw = f.readlines()
    return list(
        filter(
            lambda x: x[:6] == "ATOM  " or x[:6] == "HETATM",
            map(lambda x: x.strip(), raw),
        )
    )


class Receptor:
    @classmethod
    def from_pdb_file(cls, file_path, pdbid=""):
        pdb_atom_lines = _get_pdb_atom_lines(file_path)
        result = []
        for atomline in pdb_atom_lines:
            d = {
                "atom_name": atomline[12:16].strip(),
                "res_name": atomline[17:20].strip(),
                "ch_name": atomline[21:22].strip(),
                "res_id": int(atomline[22:26]),
                "coord": (
                    float(atomline[30:38]),  # x coord
                    float(atomline[38:46]),  # y coord
                    float(atomline[46:54]),  # z coord
                ),
            }
            result.append(d)
        return cls(result, pdbid)

    @classmethod
    def from_list(cls, d):
        return cls(d)

    def __init__(self, data, pdbid=""):
        self.data = data
        self.pdbid = pdbid

    def __len__(self):
        return len(self.data)

    def remove_non_prot(self):
        return self.filter_atoms(
            lambda d: not (
                (d["atom_name"] in atom_names) and (d["res_name"] in res_names)
            )
        )

    def limit_to_ca(self):
        return self.filter_atoms(lambda d: d["atom_name"] != "CA")

    def filter_atoms(self, fn):
        new_data = []
        for d in self.data:
            if fn(d):
                continue
            new_data.append(d)
        return Receptor(new_data)

    def limit_to_near(self, crds, dist_max):
        new_data = []
        for d in self.data:
            xyz = np.asarray(d["coord"])
            delt = xyz - crds
            dist = np.sqrt(np.sum(delt * delt, axis=1))
            dist = np.min(dist)
            if dist >= dist_max:
                continue
            d["min_dist"] = float(dist)
            new_data.append(d)
        return Receptor(new_data)

    def get_atom_feats(self):
        for d in self.data:
            yield ProtAtomFeature(d["res_name"], d["atom_name"])

    def get_aa(self, ind):
        return self.data[ind]["res_name"]

    def get_aa_list(self):
        for d in self.data:
            yield d["res_name"]

    def get_coord(self, ind):
        return self.data[ind]["coord"]

    def get_np_coord(self, dtype=np.float32):
        return np.asarray([d["coord"] for d in self.data], dtype=dtype)

    def to_list(self):
        return self.data

    def to_pdb_file(self, fn):
        with open(fn, "w") as f:
            seq = 1
            for d in self.data:
                aname = d["atom_name"]
                resn = d["res_name"]
                chnam = d["ch_name"]
                resid = d["res_id"]
                x, y, z = d["coord"]
                o, b = 0, 0
                f.write(
                    f"ATOM  {seq:5d} {aname:^4s} {resn:3s} {chnam:1s}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{o:6.2f}{b:6.2f}          \n"  # NOQA
                )
                seq += 1
