from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .cc_nissm import CCNISSMModel


class TFTInventoryModel(CCNISSMModel):
    model_key = "tft_inventory"
    model_family = "tft_inventory"
    model_version = "tft-inventory-bootstrap-v1"
    method = "tft_inventory_flow_fallback"

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> "TFTInventoryModel":
        super().fit(train_df, valid_df)
        return self

    def predict_distribution(self, rows: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
        return super().predict_distribution(rows, debug=debug)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TFTInventoryModel":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Artifact is not {cls.__name__}")
        return obj
