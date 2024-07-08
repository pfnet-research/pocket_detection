import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from pokeformer.config_wrapper import load_class, obj_from_config
from pokeformer.models.base_model import BaseModel
from pokeformer.models.metrics_wrapper import MetricsWrapper
from torch import nn
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryAveragePrecision, BinaryF1Score)

logger = logging.getLogger(__name__)


@dataclass
class PredictorConfig(BaseModel._Config):
    model: Any
    focal_gamma: Optional[float] = None


class Predictor(BaseModel):
    @classmethod
    def get_config_class(cls):
        return PredictorConfig

    def __init__(self, config, vocab_dict):
        super().__init__(None)

        vocab_size = len(vocab_dict["prot_vocab"])
        config["model"]["vocab_size"] = vocab_size
        model, res_cfg = obj_from_config(load_class, config["model"])
        self.result_config = res_cfg
        self.model = model

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        logit = self.model(batch=batch)

        label = batch["y"]
        loss = self.criterion(logit, label)

        preds = torch.sigmoid(logit)
        intlabels = label.to(torch.int).detach()
        preds = preds.detach()

        results = {
            "loss": loss.item(),
            "accuracy": MetricsWrapper(BinaryAccuracy, preds=preds, target=intlabels),
            "roc_auc": MetricsWrapper(BinaryAUROC, preds=preds, target=intlabels),
            "prc_auc": MetricsWrapper(
                BinaryAveragePrecision, preds=preds, target=intlabels
            ),
            "f1score": MetricsWrapper(BinaryF1Score, preds=preds, target=intlabels),
            "preds": preds,
        }

        total_loss = loss
        results["total_loss"] = total_loss

        return results

    def get_metrics_keys(self):
        return ["loss", "accuracy", "roc_auc", "prc_auc", "f1score"]

    def generate(self, *args, **kwargs):
        raise NotImplementedError()

    def reconstruct(self, batch, temp=None, max_len=100):
        raise NotImplementedError()

    @property
    def latent_size(self):
        return 0
