# DG-NISSM v2: CDG-NMIP

`dg_nissm` is implemented as a censored dynamic-graph neural marked inventory process.
It predicts four attempted flow distributions over the requested horizon:

- docked eBike departures
- docked eBike arrivals
- classic-bike departures
- classic-bike arrivals

Those intensities are decoded through the finite-capacity inventory DP in
`divvy.inventory_dp`, so the final PMFs satisfy `0 <= E <= Q <= capacity`.

## Offline Training

Use offline training only:

```bash
uv run python -m divvy.train_sota dg-nissm \
  --history-hours 24 \
  --valid-hours 4 \
  --anchor-every-min 2 \
  --horizons 5 10 15 20 \
  --device auto \
  --epochs 8 \
  --batch-size 4096 \
  --max-examples 600000 \
  --register \
  --benchmark-runtime
```

`train_sota all` attempts DG-NISSM and registers it only when the fit produces a
real artifact and passes output/PMF quality gates. If minimum data thresholds are
not met, the result is skipped and no fake DG-NISSM artifact is saved.

## Runtime Policy

API and dashboard scoring load DG-NISSM only from `model_registry`. If no trained
artifact is available, `dg_nissm` is marked unusable and existing fallback models
continue to serve predictions. `predict_distribution` never queries the DB and
never trains.

## Leakage Policy

Feature construction uses backward-as-of current state. Shifted station priors
are chronological expanding statistics with the current row shifted out. In the
offline train/validation split, validation priors are initialized from training
history and then updated walk-forward by earlier validation anchors only.

Sequence features, when present, are built from observations at or before the
anchor. Runtime rows may omit sequences; the model then uses a trend-based
aggregate fallback and exposes the same inventory-consistent output contract.

## Artifact Contents

The pickled model stores CPU tensors and metadata:

- `CDGNMIPConfig`
- PyTorch `state_dict`
- feature columns and scaler
- station index map
- graph cache edge lists
- zero calibrator
- validation/training metrics
- method `dg_nissm_cdg_nmip_trained_v1`
