import pandas as pd
import torch


class MetricsWrapper:
    ndim = 0

    def merged_state(self, met2):
        preds = self.preds + met2.preds
        target = self.target + met2.target
        assert self.metrics == met2.metrics
        return MetricsWrapper(self.metrics, preds, target, save_csv=self.save_csv)

    def __init__(self, metrics, preds, target, save_csv=None):
        self.save_csv = save_csv
        self.result = None
        self.metrics = metrics

        if isinstance(preds, list):
            self.preds = preds
        else:
            self.preds = [preds]

        if isinstance(target, list):
            self.target = target
        else:
            self.target = [target]

    def __add__(self, other):
        self.result = None
        if isinstance(other, MetricsWrapper):
            return self.merged_state(other)
        else:
            return self

    def __radd__(self, other):
        self.result = None
        if isinstance(other, MetricsWrapper):
            return self.merged_state(other)
        else:
            return self

    def __mul__(self, other):
        self.result = None
        return self

    def __rmul__(self, other):
        self.result = None
        return self

    def __truediv__(self, other):
        self.result = None
        return self

    def __rtruediv__(self, other):
        self.result = None
        return self

    def __float__(self) -> float:
        if self.result is None:
            preds = torch.cat(self.preds).cpu()
            target = torch.cat(self.target).cpu()

            result = self.metrics()(preds=preds, target=target)
            self.result = float(result.item())

            if self.save_csv is not None:
                df = pd.DataFrame({"pred": preds, "label": target})
                df.to_csv(self.save_csv)
        return self.result
