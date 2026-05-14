from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .cc_nissm import CCNISSMModel


class STGNCDEInventoryModel(CCNISSMModel):
    model_key = "stg_ncde_inventory"
    model_family = "stg_ncde_inventory"
    model_version = "stg-ncde-inventory-bootstrap-v1"
    method = "stg_ncde_inventory_flow_fallback"

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> "STGNCDEInventoryModel":
        super().fit(train_df, valid_df)
        return self

    def predict_distribution(self, rows: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
        return super().predict_distribution(rows, debug=debug)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "STGNCDEInventoryModel":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Artifact is not {cls.__name__}")
        return obj
