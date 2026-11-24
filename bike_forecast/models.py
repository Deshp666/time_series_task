"""Adapters around the three Nixtla forecasting frameworks."""
from __future__ import annotations

import gc
import time
import numpy as np

from .data import design_frames

ML = {"Ridge", "RandomForestRegressor", "HistGradientBoostingRegressor"}
DL = {"MLP", "LSTM", "NHITS"}


def group(name):
    return "ML" if name in ML else "DL" if name in DL else "baseline" if name == "SeasonalNaive" else "statistical"


def cuda_info():
    try:
        import torch
        return {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}
    except ImportError:
        return {"cuda_available": False, "reason": "torch is not installed"}


def forecast(train, name, params, h, seed=42, options=None, dl_config=None, smoke=False, threads=4):
    options = options or {}
    x, future, features = design_frames(train, h, options.get("calendar", True), options.get("weather", False))
    if options.get("log1p", False):
        x["y"] = np.log1p(x.y)
    family = group(name)
    peak = None
    setup_start = time.perf_counter()
    if family in {"statistical", "baseline"}:
        from statsforecast import StatsForecast
        from statsforecast import models as sm
        constructors = {"ETS": sm.AutoETS, "CES": sm.AutoCES}
        if name == "MSTL":
            model = sm.MSTL(**params, trend_forecaster=sm.AutoETS(model="ZZN"), alias=name)
        else:
            cls = constructors[name] if name in constructors else getattr(sm, name)
            model = cls(**params, alias=name)
        engine = StatsForecast(models=[model], freq="h", n_jobs=1)
        t0 = time.perf_counter()
        engine.fit(x[["unique_id", "ds", "y"]])
        t1 = time.perf_counter()
        result = engine.predict(h=h)[name].to_numpy()
    elif family == "ML":
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import RollingMean, RollingStd
        if name == "Ridge":
            estimator = make_pipeline(StandardScaler(), Ridge(**params))
        elif name == "RandomForestRegressor":
            estimator = RandomForestRegressor(**params, random_state=seed, n_jobs=threads)
        else:
            # Avoid random internal validation: tuning is explicitly temporal.
            estimator = HistGradientBoostingRegressor(**params, random_state=seed, early_stopping=False)
        engine = MLForecast(models={name: estimator}, freq="h", lags=[1, 2, 24, 48, 168, 336],
                            lag_transforms={1: [RollingMean(24), RollingStd(24), RollingMean(168), RollingStd(168)]},
                            num_threads=threads)
        t0 = time.perf_counter()
        engine.fit(x, static_features=[])
        t1 = time.perf_counter()
        result = engine.predict(h=h, X_df=future if features else None)[name].to_numpy()
    else:
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast import models as nm
        from neuralforecast.losses.pytorch import MAE
        has_cuda = torch.cuda.is_available()
        if not has_cuda and not smoke:
            raise RuntimeError("Full DL requires CUDA. CPU fallback is intentionally disabled.")
        torch.set_num_threads(threads)
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
        settings = dict(dl_config or {})
        if smoke:
            settings.update(max_steps=5, val_check_steps=1, early_stop_patience_steps=2, windows_batch_size=16)
        architecture = {}
        if name == "MLP":
            architecture = {"hidden_size": 64, "num_layers": 2}
        elif name == "LSTM":
            architecture = {"encoder_hidden_size": 32, "encoder_n_layers": 1, "decoder_hidden_size": 32}
        elif name == "NHITS":
            architecture = {"mlp_units": [[64, 64], [64, 64], [64, 64]]}
        model = getattr(nm, name)(h=h, **params, **settings, **architecture, loss=MAE(),
            scaler_type="standard", futr_exog_list=features or None, random_seed=seed,
            accelerator="gpu" if has_cuda else "cpu", devices=1,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, alias=name)
        engine = NeuralForecast(models=[model], freq="h")
        t0 = time.perf_counter()
        engine.fit(df=x, val_size=max(168, h))
        if has_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        result = engine.predict(futr_df=future if features else None)[name].to_numpy()
        if has_cuda:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
    t2 = time.perf_counter()
    result = np.asarray(result, float).reshape(-1)
    if options.get("log1p", False):
        result = np.expm1(result)
    if len(result) != h or not np.isfinite(result).all():
        raise ValueError("Invalid forecast shape or non-finite values")
    # Counts cannot be negative. Apply identically to every model and retain floating point predictions.
    result = np.maximum(result, 0)
    del engine
    gc.collect()
    return result, {"setup_seconds": t0-setup_start, "fit_seconds": t1-t0,
                    "predict_seconds": t2-t1, "peak_vram_bytes": peak}
