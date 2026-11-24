"""Isolated, resumable jobs. Each process is bounded by the remaining run budget."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import ROOT, load_data, training_frame
from .evaluation import metrics
from .models import DL, ML, cuda_info, forecast, group


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(path)


def fingerprint(config):
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode())
    digest.update((ROOT / "data/processed/hourly.csv").read_bytes())
    for path in sorted((ROOT / "bike_forecast").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(environment(), sort_keys=True).encode())
    return digest.hexdigest()[:20]


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def worker(job_path):
    path = Path(job_path)
    job = json.loads(path.read_text(encoding="utf-8"))
    try:
        df = load_data()
        options = job.get("options", {})
        train = training_frame(df, job["origin"], options.get("missing", 0), job["seed"])
        if job["profile"] == "smoke":
            train = train.tail(24*45).reset_index(drop=True)
        pred, timing = forecast(train, job["model"], job["params"], job["horizon"], job["seed"],
                                options, job["dl"], job["profile"] == "smoke", job["threads"])
        dates = pd.date_range(job["origin"], periods=job["horizon"], freq="h")
        actual = df.set_index("ds").reindex(dates)
        observed = actual.observed.fillna(False).to_numpy(dtype=bool)
        scores = metrics(actual.y, pred, observed, train.y)
        rows = pd.DataFrame({"ds": dates, "y": actual.y.to_numpy(), "observed": observed, "prediction": pred})
        rows.to_csv(path.parent / "predictions.csv", index=False)
        write_json(path.parent / "result.json", {**job, **timing, **scores, "status": "ok", "training_rows": len(train)})
    except Exception as exc:
        write_json(path.parent / "result.json", {**job, "status": "error", "error": str(exc), "traceback": traceback.format_exc()})
        raise


class Run:
    def __init__(self, profile):
        self.config = yaml.safe_load((ROOT / "configs/study.yaml").read_text(encoding="utf-8"))
        self.profile = profile
        if profile == "smoke":
            self.config["horizons"] = [24]
            self.config["splits"] = {k: v[:1] for k, v in self.config["splits"].items()}
            self.config["models"] = {k: v[:1] for k, v in self.config["models"].items()}
            self.config["budget_seconds"] = 1200
            self.config["job_timeout_seconds"] = 180
        self.config["profile"] = profile
        self.id = fingerprint(self.config)
        self.folder = ROOT / "artifacts" / f"{profile}-{self.id}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.deadline = time.monotonic() + self.config["budget_seconds"]
        self.cache_hits = 0

    def job(self, model, params, horizon, phase, origin, options=None):
        options = options or {}
        job = {"model": model, "group": group(model), "params": params, "horizon": horizon,
               "phase": phase, "origin": origin, "seed": options.get("seed", self.config["seed"]),
               "options": options, "profile": self.profile, "dl": self.config["dl"], "threads": self.config["threads"]}
        key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
        job["job_id"] = key
        directory = self.folder / "jobs" / key
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["status"] == "ok" and (directory / "predictions.csv").exists():
                self.cache_hits += 1
                self.records.append(result)
                return result
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            result = {**job, "status": "budget_exhausted"}
        elif model in DL and importlib.util.find_spec("neuralforecast") is None:
            result = {**job, "status": "missing_dependency", "error": "Install project [dl] extra"}
        else:
            write_json(directory / "job.json", job)
            # Never mistake a stale successful result for a new worker's output.
            write_json(result_path, {**job, "status": "running"})
            print(f"{phase}: {model} h={horizon} {origin} {options.get('name', '')}", flush=True)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS=str(self.config["threads"]), MKL_NUM_THREADS=str(self.config["threads"]),
                       PYTHONIOENCODING="utf-8", NUMBA_NUM_THREADS=str(self.config["threads"]))
            try:
                with (directory / "process.log").open("w", encoding="utf-8") as log:
                    process = subprocess.run([sys.executable, "-m", "bike_forecast", "_worker", str(directory / "job.json")],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        timeout=min(remaining, self.config["job_timeout_seconds"]), check=False)
                result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
                    **job, "status": "error", "error": f"Worker exit {process.returncode}; inspect process.log"}
            except subprocess.TimeoutExpired:
                result = {**job, "status": "timeout", "error": "Per-job or total budget exceeded"}
        write_json(result_path, result)
        self.records.append(result)
        return result

    def execute(self):
        write_json(self.folder / "environment.json", environment())
        hardware = cuda_info()
        write_json(self.folder / "hardware.json", hardware)
        (self.folder / "resolved.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        write_json(ROOT / "artifacts" / f"latest_{self.profile}.json", {"path": self.folder.relative_to(ROOT).as_posix()})
        if self.profile == "full" and not hardware["cuda_available"]:
            write_json(self.folder / "status.json", {"complete": False, "profile": self.profile,
                "reason": "CUDA unavailable: full run blocked before expensive training", "hardware": hardware})
            raise RuntimeError("Full run requires the target NVIDIA machine and CUDA-enabled PyTorch. See hardware.json.")
        write_json(self.folder / "status.json", {"complete": False, "course_complete": False,
            "profile": self.profile, "reason": "Run in progress; resume with the same command after interruption"})
        selected = {}
        selection_scores = {}
        for h in self.config["horizons"]:
            selected[str(h)] = {}
            for model, configs in self.config["models"].items():
                candidates = []
                for params in configs:
                    results = [self.job(model, params, h, "validation", origin) for origin in self.config["splits"]["validation"]]
                    if all(r["status"] == "ok" for r in results):
                        candidates.append((float(np.mean([r["MAE"] for r in results])), params))
                if candidates:
                    score, params = min(candidates, key=lambda p: p[0])
                    selected[str(h)][model] = params
                    selection_scores[(h, model)] = score
        write_json(self.folder / "selected.json", selected)
        # Selection is fixed before either calibration or test is accessed.
        for phase in ("calibration", "test"):
            for h in self.config["horizons"]:
                for model, params in selected[str(h)].items():
                    for origin in self.config["splits"][phase]:
                        self.job(model, params, h, phase, origin)
        extension_leaders = []
        if self.profile == "full":
            for family in (ML, DL):
                eligible = [(selection_scores[(24, m)], m) for m in family if (24, m) in selection_scores]
                if eligible:
                    model = min(eligible)[1]
                    extension_leaders.append(model)
                    for options in self.config["extensions"]:
                        for origin in self.config["splits"]["validation"]:
                            self.job(model, selected["24"][model], 24, "extension", origin, options)
        write_json(self.folder / "extension_leaders.json", extension_leaders)
        write_json(self.folder / "records.json", self.records)
        failures = [r for r in self.records if r["status"] != "ok"]
        write_json(self.folder / "errors.json", failures)
        # Every planned configuration/window must finish, not just the selected winners.
        models_ok = all(len(selected[str(h)]) == len(self.config["models"]) for h in self.config["horizons"])
        complete = not failures and models_ok and (self.profile == "smoke" or len(extension_leaders) == 2)
        write_json(self.folder / "status.json", {"complete": complete, "course_complete": complete and self.profile == "full",
            "profile": self.profile, "jobs": len(self.records), "failed": len(failures), "cache_hits": self.cache_hits,
            "elapsed_seconds": self.config["budget_seconds"] - max(0, self.deadline-time.monotonic()),
            "reason": "All scheduled jobs finished" if complete else "Incomplete jobs or missing model families; see errors.json"})
        return self.folder
