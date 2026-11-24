"""Build a presentation notebook from saved forecasting results."""
import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def code(source):
    return nbformat.v4.new_code_cell(dedent(source).strip())


def build_notebook(profile):
    markdown = nbformat.v4.new_markdown_cell
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python (project environment)", "language": "python", "name": "python3",
    }
    notebook.cells = [
        markdown(
            "# Прогноз почасового спроса на велопрокат\n\n"
            "Capital Bikeshare, 2011–2012 годы. Задача — оценить число аренд в каждый час следующих суток "
            "или недели. Последовательно покажите данные, план проверки, сравнение моделей и надёжность прогноза. "
            "Ячейки читают сохранённые результаты. [Описание проекта](docs/README.md)"
        ),
        code(f"""
            from pathlib import Path, PureWindowsPath
            import json
            import pandas as pd
            import yaml
            from IPython.display import display, Markdown, Image

            PROFILE = {profile!r}
            cwd = Path.cwd()
            root = next(path for path in (cwd, *cwd.parents) if (path / "configs/study.yaml").is_file())
            pointer = json.loads((root / "artifacts" / f"latest_{{PROFILE}}.json").read_text(encoding="utf-8"))
            folder = root.joinpath(*PureWindowsPath(pointer["path"]).parts)
            status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
            config = yaml.safe_load((folder / "resolved.yaml").read_text(encoding="utf-8"))

            def show_figure(name):
                path = folder / "figures" / name
                if path.is_file():
                    display(Image(filename=str(path)))

            if PROFILE == "smoke":
                message = "Короткая проверка на одном окне каждого этапа."
            elif status.get("course_complete"):
                message = "Полное исследование завершено."
            else:
                message = "Полное исследование продолжается. Ниже доступны EDA и сохранённые результаты валидации."
            display(Markdown(f"**Статус прогона** · {{message}}"))
        """),
        markdown(
            "## 1. Данные и предварительный анализ\n\n"
            "В строке один час. Прогнозируем столбец `y` — общее число аренд. "
            "После восстановления часовой сетки сохраняем признак наличия факта `observed`. "
            "Посмотрите на утренний и вечерний пики и пики автокорреляции на лагах 24 и 168. "
            "Они объясняют, почему проверяем суточную и недельную сезонности."
        ),
        code("""
            hourly = pd.read_csv(root / "data/processed/hourly.csv")
            display(hourly[["ds", "y", "observed"]].head())
            print(f"Часов в сетке: {len(hourly)}. Наблюдаемых: {hourly.observed.sum()}. "
                  f"Пропущено: {(~hourly.observed).sum()}.")
            for name in ("eda.png", "acf_pacf.png", "decomposition.png"):
                show_figure(name)
        """),
        markdown(
            "## 2. Временная проверка и статистические модели\n\n"
            "Каждая дата ниже — первая точка прогноза. Для неё модель обучается на всех доступных "
            "часах до начала окна в полном профиле. Валидация нужна, чтобы выбрать настройки, "
            "например сезонность 24 или 168 часов. Ноябрьские прогнозы используются для калибровки "
            "интервалов, декабрьские — для итоговой проверки.\n\n"
            "**Бейзлайн** берёт спрос того же часа неделю назад. **Статистические методы** — ARIMA, ETS, "
            "Theta, CES и MSTL. Сравниваем также AutoARIMA и AutoETS с ручными вариантами.\n\n"
            "**MAE** показывает средний размер промаха в арендах за час. Меньшее число соответствует "
            "меньшей ошибке на одинаковых окнах."
        ),
        code("""
            phase_labels = {"validation": "Валидация", "calibration": "Калибровка", "test": "Тест"}
            display(pd.DataFrame({phase_labels[phase]: origins for phase, origins in config["splits"].items()}))

            summary_path = folder / "summary.csv"
            if summary_path.is_file():
                scores = pd.read_csv(summary_path)
            else:
                scores = pd.DataFrame()
                selected_path = folder / "selected.json"
                if selected_path.is_file():
                    selected = json.loads(selected_path.read_text(encoding="utf-8"))
                    validation = []
                    for path in (folder / "jobs").glob("*/result.json"):
                        row = json.loads(path.read_text(encoding="utf-8"))
                        setting = selected.get(str(row["horizon"]), {}).get(row["model"])
                        if row["status"] == "ok" and row["phase"] == "validation" and row["params"] == setting:
                            validation.append(row)
                    if validation:
                        scores = (pd.DataFrame(validation)
                                  .groupby(["phase", "group", "model", "horizon"], as_index=False)
                                  [["MAE", "fit_seconds", "predict_seconds"]].mean())

            def show_scores(groups):
                if scores.empty:
                    print("Сводка появится после подбора настроек.")
                    return
                part = scores[scores.group.isin(groups) & scores.phase.isin(["validation", "test"])]
                columns = {"phase": "Этап", "horizon": "Горизонт, ч", "model": "Модель",
                           "MAE": "MAE", "fit_seconds": "Обучение, с", "predict_seconds": "Прогноз, с"}
                part = part.sort_values(["horizon", "phase", "MAE"])[list(columns)].round(3).rename(columns=columns)
                part["Этап"] = part["Этап"].map(phase_labels)
                display(part.reset_index(drop=True))

            show_scores(["baseline", "statistical"])
        """),
        markdown(
            "## 3. Машинное и глубокое обучение\n\n"
            "**Машинное обучение** — Ridge, случайный лес и градиентный бустинг. Модели используют "
            "прошлый спрос, скользящие статистики и календарь. **Нейросети** — MLP, LSTM и NHITS. "
            "У них сравниваются входные истории разной длины. Результаты ниже помогают понять, "
            "где выигрыш в точности оправдывает время обучения."
        ),
        code('show_scores(["ML", "DL"])'),
        markdown(
            "## 4. Итоговая точность, интервалы и пайплайн\n\n"
            "Подготовка данных → подбор настроек → калибровка → итоговая проверка → отчёт. "
            "Сравните MAE выбранной на валидации модели с недельным бейзлайном. "
            "Полоса на графике — прогнозный интервал. В таблице приведены его фактическое покрытие и ширина."
        ),
        code("""
            coverage_path = folder / "coverage_by_window.csv"
            if scores.empty or "test" not in set(scores.phase) or not coverage_path.is_file():
                print("Декабрьские метрики и прогнозные интервалы появятся после полного прогона.")
            else:
                coverage = pd.read_csv(coverage_path)
                comparison = []
                for h, validation in scores[scores.phase == "validation"].groupby("horizon"):
                    leader = validation.sort_values(["MAE", "model"]).iloc[0].model
                    tested = scores[(scores.horizon == h) & (scores.phase == "test")].set_index("model")
                    if leader not in tested.index or "SeasonalNaive" not in tested.index:
                        continue
                    intervals = coverage[(coverage.horizon == h) & (coverage.model == leader)
                                         & (coverage.level == 0.95)]
                    comparison.append({"Горизонт, ч": h, "Выбор по валидации": leader,
                                       "MAE модели на тесте": tested.loc[leader, "MAE"],
                                       "MAE бейзлайна на тесте": tested.loc["SeasonalNaive", "MAE"],
                                       "Покрытие 95%-го интервала, %": 100 * intervals.coverage.mean(),
                                       "Средняя ширина": intervals.mean_width.mean()})
                if comparison:
                    display(pd.DataFrame(comparison).round(2))
                    for row in comparison:
                        h = row["Горизонт, ч"]
                        show_figure(f"forecast_{row['Выбор по валидации']}_{h}.png")
            if (folder / "REPORT.md").is_file():
                report = (folder / "REPORT.md").relative_to(root).as_posix()
                display(Markdown(f"[Полный отчёт этого прогона]({report})"))
        """),
    ]
    return notebook


def execute_notebook(notebook):
    for variable, directory in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter")):
        if variable not in os.environ:
            cache_path = ROOT / ".cache" / directory
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(cache_path)
    client = NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(ROOT)}})
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.execute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--execute", action="store_true", help="Embed outputs from saved artifacts")
    args = parser.parse_args()
    notebook = build_notebook(args.profile)
    if args.execute:
        execute_notebook(notebook)
    path = ROOT / f"study_{args.profile}.ipynb"
    nbformat.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
