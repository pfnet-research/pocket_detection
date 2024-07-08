import logging

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def make_split_indices(self, train_size, valid_size, recon_size):
        nmol = len(self)
        assert train_size + valid_size == nmol
        perm = np.random.permutation(nmol)
        train_mol_ids = perm[:train_size]
        valid_mol_ids = perm[train_size:]
        if valid_size >= recon_size:
            recon_mol_ids = np.random.choice(valid_mol_ids, recon_size, replace=False)
        elif train_size >= recon_size:
            recon_mol_ids = np.random.choice(train_mol_ids, recon_size, replace=False)
        else:
            raise RuntimeError(
                "invalid train/valid/recon size: "
                f"{train_size}/{valid_size}/{recon_size}"
            )

        logger.info(f"train ids: {train_mol_ids}")
        logger.info(f"valid ids: {valid_mol_ids}")
        logger.info(f"recon ids: {recon_mol_ids}")
        return train_mol_ids, valid_mol_ids, recon_mol_ids

    def get_metrics_keys(self):
        return ["recon/valid_ratio", "recon/ave_sim"]

    def get_vocab_dict(self):
        raise NotImplementedError()
