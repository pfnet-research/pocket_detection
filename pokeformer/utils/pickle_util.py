import logging
import pickle
from pathlib import Path
from urllib.parse import urlparse

import pfio
import torch
import tqdm

logger = logging.getLogger(__name__)


def write_list(data, f, use_prog_bar=True):
    nlen = len(data)
    pickle.dump(nlen, f)
    if use_prog_bar:
        for i in tqdm.tqdm(data):
            pickle.dump(i, f)
    else:
        for i in data:
            pickle.dump(i, f)


def write_list_iter(f, gen, nlen=-1):
    pickle.dump(nlen, f)
    for i in gen:
        pickle.dump(i, f)


def read_list(f, use_prog_bar=True, length=None):
    nlen = pickle.load(f)
    if length is not None:
        logger.info(f"file length {nlen} is overwritten by arg {length=}")
        nlen = length
    result = []
    if use_prog_bar:
        for _ in tqdm.trange(nlen):
            try:
                d = pickle.load(f)
            except EOFError:
                break
            result.append(d)
    else:
        for _ in range(nlen):
            try:
                d = pickle.load(f)
            except EOFError:
                break
            result.append(d)

    return result


def read_list_iter(f, use_prog_bar=True):
    nlen = pickle.load(f)
    rngfn = range
    if use_prog_bar:
        rngfn = tqdm.trange
    for _ in rngfn(nlen):
        try:
            d = pickle.load(f)
        except EOFError:
            break
        yield d


def conv_path(in_path):
    urlobj = urlparse(in_path)
    if urlobj.scheme == "":
        path_obj = Path(in_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        fs = pfio.v2.from_url(f"{urlobj.scheme}://{urlobj.netloc}")
        path_obj = pfio.v2.pathlib.Path(urlobj.path, fs=fs)
    return path_obj


def write_data(out_path_str, out_dat):
    out_path = conv_path(out_path_str)

    with out_path.open("wb") as f:
        if out_path.suffix == ".pklstr":
            write_list(out_dat, f)
        elif out_path.suffix == ".pkl":
            pickle.dump(out_dat, f)
        elif out_path.suffix == ".pt":
            torch.save(out_dat, f)
        else:
            raise ValueError(f"uknown format ext: {out_path}")


def read_data(in_path_str, length=None):
    in_path = conv_path(in_path_str)

    with in_path.open("rb") as f:
        if in_path.suffix == ".pklstr":
            return read_list(f, length=length)
        elif in_path.suffix == ".pkl":
            return pickle.load(f)
        elif in_path.suffix == ".pt":
            return torch.load(f)
        else:
            raise ValueError(f"uknown format ext: {in_path}")
