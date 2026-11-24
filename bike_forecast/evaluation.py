"""Metrics, calibration and paired time-block uncertainty."""
import math
import numpy as np


def metrics(actual, predicted, observed, train_y):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mask = np.asarray(observed, bool) & np.isfinite(actual)
    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite forecast")
    if not mask.any():
        raise ValueError("No observed test targets")
    a, p = actual[mask], predicted[mask]
    errors = np.abs(a-p)
    train_y = np.asarray(train_y, float)
    scale = np.mean(np.abs(train_y[168:] - train_y[:-168])) if len(train_y) > 168 else np.nan
    denominator = np.abs(a).sum()
    return {"MAE": float(errors.mean()), "RMSE": float(np.sqrt(np.mean((a-p)**2))),
            "WAPE": float(errors.sum()/denominator) if denominator else None,
            "MASE": float(errors.mean()/scale) if np.isfinite(scale) and scale > 0 else None,
            "n_observed": int(mask.sum())}


def calibration_radius(errors, level):
    errors = np.asarray(errors, float)
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        raise ValueError("Empty calibration errors")
    rank = min(len(errors), math.ceil((len(errors)+1)*level))
    return float(np.sort(errors)[rank-1])


def paired_block_bootstrap(windows, seed=42, repetitions=1000):
    """Each window contains aligned loss differences; never bridge gaps between origins."""
    blocks = [np.asarray(w[i:i+24], float) for w in windows for i in range(0, len(w), 24)]
    blocks = [b[np.isfinite(b)] for b in blocks if np.isfinite(b).any()]
    if not blocks:
        raise ValueError("No paired observed errors")
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean()
             for _ in range(repetitions)]
    return {"mean_loss_difference": float(np.concatenate(blocks).mean()),
            "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
            "blocks": len(blocks), "repetitions": repetitions}
