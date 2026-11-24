"""Generate the Russian report exclusively from recorded observations and runs."""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from .data import ROOT, SOURCE, WEATHER, load_data, training_frame
from .evaluation import calibration_radius, paired_block_bootstrap
from .runner import write_json


def figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"![{name}](figures/{name})"


def eda(folder):
    df = load_data()
    study = df[df.ds < "2012-11-01"].copy()
    train = training_frame(df, "2012-11-01")
    y = train.set_index("ds").y
    stats = {"grid_rows": len(df), "observed_rows": int(df.observed.sum()),
             "missing_hours": int((~df.observed).sum()), "eda_observations": int(study.observed.sum()),
             "eda_mean": float(study.y.mean()), "eda_std": float(study.y.std()),
             "eda_max": float(study.y.max()), "start": str(df.ds.min()), "end": str(df.ds.max())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, values in (("raw", y), ("diff24", y.diff(24).dropna())):
            stats[f"adf_p_{name}"] = float(adfuller(values, maxlag=24, autolag="AIC")[1])
            stats[f"kpss_p_{name}"] = float(kpss(values, regression="c", nlags="auto")[1])
    write_json(folder.parent / "eda.json", stats)
    images = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    study.set_index("ds").y.resample("D").mean().plot(ax=axes[0, 0], title="Среднее число аренд за час по дням")
    study.y.hist(ax=axes[0, 1], bins=50)
    axes[0, 1].set_title("Распределение наблюдаемого спроса")
    study.groupby(study.ds.dt.hour).y.mean().plot(ax=axes[1, 0], title="Суточный профиль")
    study.groupby(study.ds.dt.dayofweek).y.mean().plot.bar(ax=axes[1, 1], title="Недельный профиль (0 = пн)")
    images.append(figure(fig, folder, "eda.png"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(y, lags=168, ax=axes[0], fft=True)
    plot_pacf(y, lags=48, ax=axes[1], method="ywm")
    images.append(figure(fig, folder, "acf_pacf.png"))
    decomposition = STL(y, period=24, robust=True).fit()
    images.append(figure(decomposition.plot(), folder, "decomposition.png"))
    corr = study[["y", *WEATHER]].corr()
    corr.to_csv(folder.parent / "weather_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(img, ax=ax)
    images.append(figure(fig, folder, "correlations.png"))
    return stats, images


def build_report(profile="full"):
    pointer = ROOT / "artifacts" / f"latest_{profile}.json"
    if not pointer.exists():
        raise FileNotFoundError(f"Run --profile {profile} first")
    folder = ROOT / json.loads(pointer.read_text(encoding="utf-8"))["path"]
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    records_path = folder / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    stats, images = eda(figures)
    good = [r for r in records if r["status"] == "ok"]
    job_paths = [folder / "jobs" / record["job_id"] / "predictions.csv" for record in good]
    if any(not path.is_file() for path in job_paths):
        predictions = pd.read_csv(folder / "predictions.csv", parse_dates=["ds"])
    else:
        chunks = []
        for record, path in zip(good, job_paths):
            p = pd.read_csv(path, parse_dates=["ds"])
            for field in ("model", "group", "horizon", "phase", "origin", "job_id"):
                p[field] = record[field]
            p["variant"] = record["options"].get("name", "default")
            chunks.append(p)
        predictions = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    scores = pd.DataFrame([{k: v for k, v in r.items() if k not in ("dl",)} for r in good])
    scores.to_csv(folder / "metrics_by_window.csv", index=False)
    sections = ["# Прогнозирование почасового спроса на велопрокат", "",
        f"Профиль: **{profile}**. Завершенность итогового проекта: **{status.get('course_complete', False)}**.",
        f"Статус запуска: {status.get('reason', 'неизвестен')}.",
        "Smoke — техническая проверка на 45 днях обучения, одном окне каждого этапа и 5 шагах DL. "
        "Ее результаты не подтверждают качество полного исследования." if profile == "smoke" else
        "Полный протокол использует расширяющиеся обучающие выборки, 4 окна каждого этапа и два горизонта.",
        "", "## 1. Данные, постановка задачи и EDA", "",
        f"Источник: [UCI Bike Sharing]({SOURCE}), DOI 10.24432/C5W894, Hadi Fanaee-T; "
        "публикация Hadi Fanaee-T и João Gama. Лицензия CC BY 4.0. Согласование с преподавателем не зафиксировано.",
        "Цель — офлайн-прогноз общего числа аренд cnt для планирования доступности велосипедов. "
        "Основной горизонт 24 часа, дополнительный 168 часов. Ряд один: глобальные модели между станциями не требуются.",
        f"Период: {stats['start']} — {stats['end']}. Наблюдений: **{stats['observed_rows']}**; "
        f"часов в регулярной сетке: **{stats['grid_rows']}**; отсутствующих часов: **{stats['missing_hours']}**.",
        "Временные метки трактуются как локальные метки источника без искусственного перевода в UTC. "
        "Неоднозначности перехода на летнее время нельзя восстановить из исходной таблицы. "
        "Пропуски не объявляются нулевыми: обучение использует прошлую неделю/последнее известное значение, "
        "оценка — только наблюдаемые цели. casual и registered исключены. "
        "Исследовательский EDA ограничен датой 2012-11-01.",
        *images,
        f"Вывод EDA: по {stats['eda_observations']} наблюдениям средний спрос составляет "
        f"{stats['eda_mean']:.2f} аренд/час, стандартное отклонение {stats['eda_std']:.2f}, максимум {stats['eda_max']:.0f}. "
        "Высокая изменчивость требует сравнения с сезонным бейзлайном.",
        f"ADF p-value: исходный ряд {stats['adf_p_raw']:.4g}, разность с лагом 24 {stats['adf_p_diff24']:.4g}. "
        f"KPSS p-value: {stats['kpss_p_raw']:.4g} и {stats['kpss_p_diff24']:.4g}. "
        "ADF проверяет нулевую гипотезу единичного корня, KPSS — стационарности; "
        "граничные p-value KPSS ограничены таблицей теста. Тесты не заменяют анализ сезонности.",
        "", "## 2. Статистические модели и оценка надежности", "",
        "Сравниваются пять семейств ARIMA, ETS, Theta, CES и MSTL, а также SeasonalNaive(168). "
        "AutoARIMA и AutoETS показаны дополнительно к ручным вариантам. "
        "ARIMA(1,1,1) с сезонной AR(1) задает компактную ручную модель; ETS(AAA) допускает "
        "аддитивные тренд и сезонность. Периоды 24/168 соответствуют суткам/неделе. "
        "MSTL использует обе сезонности и AutoETS для тренда. Выбор конфигурации — по среднему MAE валидации.",
        "Валидация: 1/15 сентября и 1/15 октября; калибровка: 1/8/15/22 ноября; "
        "тест: 1/8/15/22 декабря 2012 года. Предыдущие тестовые наблюдения становятся доступны "
        "для переобучения следующих окон, но не для настройки параметров.",
        "MAE и RMSE выражены в арендах/час; WAPE — доля, не проценты. "
        "MASE нормируется на обучающую ошибку сезонного наивного прогноза с лагом 168. "
        "Сводные показатели — среднее метрик окон, а не метрики искусственно склеенного ряда."]
    summaries = pd.DataFrame()
    diagnostics = []
    if not scores.empty:
        selected_path = folder / "selected.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
        main = scores[(scores.phase != "extension") & scores.apply(
            lambda r: r.phase != "validation" or r.params == selected.get(str(r.horizon), {}).get(r.model), axis=1)]
        summaries = main.groupby(["phase", "group", "model", "horizon"])[
            ["MAE", "RMSE", "WAPE", "MASE", "fit_seconds", "predict_seconds"]].mean().reset_index()
        summaries.to_csv(folder / "summary.csv", index=False)
        if selected_path.exists():
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            sections += ["", "Выбранные на валидации параметры:", "", "```json", json.dumps(selected, ensure_ascii=False, indent=2), "```"]
        test_summary = summaries[summaries.phase == "test"]
        sections += ["", "Тестовые результаты (только успешно выполненные модели):", "", test_summary.to_markdown(index=False, floatfmt=".3f")]
        intervals, coverage_rows = [], []
        for (name, h), test in predictions[predictions.phase == "test"].groupby(["model", "horizon"]):
            cal = predictions[(predictions.model == name) & (predictions.horizon == h) & (predictions.phase == "calibration")]
            cal = cal[cal.observed]
            if cal.empty:
                continue
            test = test.copy()
            for level in (.8, .95):
                tag = int(level*100)
                radius = calibration_radius(np.abs(cal.y-cal.prediction), level)
                test[f"lo{tag}"] = np.maximum(0, test.prediction-radius)
                test[f"hi{tag}"] = test.prediction+radius
                for origin, window in test.groupby("origin"):
                    observed = window[window.observed]
                    coverage_rows.append({"model": name, "horizon": h, "origin": origin, "level": level,
                        "radius": radius, "coverage": float(((observed.y >= observed[f"lo{tag}"]) & (observed.y <= observed[f"hi{tag}"])).mean()),
                        "mean_width": float((observed[f"hi{tag}"]-observed[f"lo{tag}"]).mean())})
            intervals.append(test)
            example = test[test.origin == test.origin.min()]
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(example.ds, example.y.where(example.observed), label="Наблюдение")
            ax.plot(example.ds, example.prediction, label=name)
            ax.fill_between(example.ds, example.lo95, example.hi95, alpha=.2, label="95%")
            ax.set_title(f"{name}, горизонт {h} ч")
            ax.legend()
            sections += ["", figure(fig, figures, f"forecast_{name}_{h}.png")]
        if intervals:
            pd.concat(intervals).to_csv(folder / "test_predictions_intervals.csv", index=False)
            coverage = pd.DataFrame(coverage_rows)
            coverage.to_csv(folder / "coverage_by_window.csv", index=False)
            sections += ["", "Покрытие и средняя ширина интервалов:", "",
                         coverage.groupby(["model", "horizon", "level"])[["coverage", "mean_width"]].mean().to_markdown(floatfmt=".3f")]
        # Leaders are selected using validation only, never by December test performance.
        for h, val in summaries[summaries.phase == "validation"].groupby("horizon"):
            leader = val.sort_values(["MAE", "model"]).iloc[0].model
            lead = predictions[(predictions.phase == "test") & (predictions.horizon == h) & (predictions.model == leader)]
            base = predictions[(predictions.phase == "test") & (predictions.horizon == h) & (predictions.model == "SeasonalNaive")]
            joined = lead.merge(base, on=["ds", "origin"], suffixes=("_lead", "_base"))
            if joined.empty:
                continue
            windows = []
            for origin, window in joined.groupby("origin"):
                loss = (window.y_lead-window.prediction_lead).abs()-(window.y_base-window.prediction_base).abs()
                loss = loss.where(window.observed_lead & window.observed_base)
                windows.append(loss.to_numpy())
                # Use the longest contiguous observed segment for residual tests.
                residual = (window.y_lead-window.prediction_lead).where(window.observed_lead)
                segments = residual.groupby(residual.isna().cumsum())
                residual = max((s.dropna() for _, s in segments), key=len)
                if len(residual) > 12:
                    lag = min(24, max(1, len(residual)//5))
                    lb = acorr_ljungbox(residual, lags=[lag], return_df=True)
                    diagnostics.append({"model": leader, "horizon": int(h), "origin": origin,
                        "n_contiguous": len(residual), "lag": lag, "ljung_box_p": float(lb.lb_pvalue.iloc[0])})
                    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                    axes[0].hist(residual, bins=15)
                    axes[0].set_title("Остатки")
                    plot_acf(residual, lags=lag, ax=axes[1], fft=True)
                    sections += ["", figure(fig, figures, f"residual_{leader}_{h}_{origin}.png")]
            boot = paired_block_bootstrap(windows)
            write_json(folder / f"bootstrap_{h}.json", {"leader": leader, **boot})
            sections += ["", f"Лидер валидации h={h}: **{leader}**. Разность MAE с бейзлайном (лидер − бейзлайн): "
                f"{boot['mean_loss_difference']:.3f}; блочный bootstrap 95%: "
                f"[{boot['ci_low']:.3f}; {boot['ci_high']:.3f}], блоков {boot['blocks']}. "
                "Отрицательная разность означает меньшую ошибку лидера."]
        pd.DataFrame(diagnostics).to_csv(folder / "residual_diagnostics.csv", index=False)
        if diagnostics:
            sections += ["", pd.DataFrame(diagnostics).to_markdown(index=False, floatfmt=".4g")]
        stat_test = test_summary[test_summary.group == "statistical"]
        if not stat_test.empty:
            best = stat_test.sort_values("MAE").iloc[0]
            sections += ["", f"Вывод раздела 2: среди выполненных статистических результатов минимальный тестовый MAE "
                f"{best.MAE:.3f} у {best.model}, h={best.horizon}. Это описательная оценка; выбор для использования делается на валидации."]
    else:
        sections += ["", "Результатов обучения нет. Численное сравнение моделей пока невозможно."]
    sections += ["", "Интервалы используют ноябрьские абсолютные ошибки, объединенные по шагам отдельно для модели и горизонта. "
        "Временная зависимость и изменение спроса нарушают предпосылку обменности: номинальное покрытие не гарантируется. "
        "Четыре декабрьских окна дают ограниченную оценку надежности. Ljung–Box здесь — диагностика ошибок многокрокового прогноза, "
        "а не доказательство корректности спецификации модели.",
        "", "## 3. ML, DL и дополнительные эксперименты", "",
        "ML использует рекурсивный прогноз с лагами 1, 2, 24, 48, 168, 336 и скользящими статистиками "
        "за 24/168 часов, сдвинутыми минимум на 1 час. Ridge включает стандартизацию признаков внутри fit. "
        "RandomForest и HistGradientBoosting моделируют нелинейные связи. "
        "MLP, LSTM и NHITS сравнивают полносвязную, рекуррентную и многомасштабную архитектуры. "
        "DL нормирует данные в рамках обучения; ранняя остановка использует последний внутренний holdout, не внешнее окно оценки.",
        "Погодная абляция использует лаг 168 часов для каждого признака: все его значения доступны при выпуске прогноза "
        "на обоих горизонтах. Это проверка полезности исторической погоды, а не эксперимент с идеальным прогнозом погоды. "
        "log1p оценивается после обратного преобразования; коррекция смещения не применяется. "
        "Отрицательные прогнозы всех моделей обрезаются до нуля без округления."]
    if not summaries.empty:
        ml_dl = summaries[(summaries.phase == "test") & summaries.group.isin(["ML", "DL"])]
        for family in ("ML", "DL"):
            subset = ml_dl[ml_dl.group == family]
            if not subset.empty:
                best = subset.sort_values("MAE").iloc[0]
                sections += ["", f"Вывод {family}: лучший наблюдаемый тестовый MAE {best.MAE:.3f} "
                    f"у {best.model}, h={best.horizon}; сравнение ограничено выполненными конфигурациями."]
            else:
                sections += ["", f"Вывод {family}: тестовых результатов пока нет."]
    if good:
        extensions = scores[scores.phase == "extension"].copy()
        if not extensions.empty:
            extensions["variant"] = extensions.options.map(lambda v: v["name"])
            ext = extensions.groupby(["model", "variant"])[["MAE", "RMSE", "fit_seconds"]].mean()
            ext.to_csv(folder / "extensions.csv")
            sections += ["", ext.to_markdown(floatfmt=".3f"), "",
                         "Сравнение расширений проводится только на валидации; исходные значения приведены в summary.csv (phase=validation)."]
        else:
            sections += ["", "Расширенные эксперименты в этом запуске не выполнены."]
    sections += ["", "## 4. Пайплайн и воспроизводимость", "",
        "Загрузка → регулярная сетка и маска → временные разбиения → подбор параметров → калибровка → тест → отчет. "
        "Код запускает эксперименты в отдельных процессах. Кэш учитывает данные, конфигурацию, код и версии окружения. "
        "Успешные задания переиспользуются; ошибочные повторяются. Подробности: resolved.yaml, environment.json, hardware.json, "
        "records.json, metrics_by_window.csv и predictions.csv.",
        f"Вывод раздела 4: записано заданий {status.get('jobs', 0)}, неуспешных {status.get('failed', 0)}, "
        f"попаданий в кэш {status.get('cache_hits', 0)}; время текущего запуска {status.get('elapsed_seconds', 0):.1f} с. "
        "Тесты корректности запускаются командой python -m pytest.",
        "", "## Общее заключение", "",
        f"Исследуется UCI Bike Sharing за 2011–2012 годы: {stats['observed_rows']} наблюдаемых часов. "
        "Задача — прогноз спроса на 24 и 168 часов с сопоставлением статистических, ML- и DL-моделей."]
    if not summaries.empty:
        for h, val in summaries[summaries.phase == "validation"].groupby("horizon"):
            leader = val.sort_values(["MAE", "model"]).iloc[0]
            test = summaries[(summaries.phase == "test") & (summaries.horizon == h) & (summaries.model == leader.model)]
            if not test.empty:
                result = test.iloc[0]
                sections += [f"Для h={h} лидер валидации {leader.model}: тестовые MAE {result.MAE:.3f}, "
                             f"RMSE {result.RMSE:.3f}, WAPE {result.WAPE:.3f}. "
                             "Оценки применимы к указанным временным окнам и профилю запуска."]
    sections += ["Полное исследование завершено." if status.get("course_complete") else
                 "Для подтверждения итоговых результатов необходимо завершить полный прогон."]
    if not predictions.empty:
        predictions.to_csv(folder / "predictions.csv", index=False)
        predictions[predictions.phase == "test"][["ds", "y", "observed", "horizon", "origin"]].drop_duplicates().to_csv(
            folder / "test_data.csv", index=False)
    path = folder / "REPORT.md"
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path
