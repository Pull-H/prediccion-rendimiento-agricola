"""Robustness analyses for completed nested temporal searches (protocol v3).

A single chronological hold-out cannot show whether a model beats the naive
yield of the previous season, whether two models differ once the dependence of
errors within a country is respected, whether accuracy survives countries never
seen in training, or whether conformal coverage recovers when intervals are
recalibrated with outcomes that are already known. This module adds those checks
without re-tuning on, or otherwise touching, the test period:

* persistence reference (last observed season of the same region), WAPE and MASE;
* paired comparisons with country-clustered inference and Holm adjustment;
* sequential split-conformal recalibration (static, expanding and sliding windows);
* spatio-temporal validation: test-period rows of countries excluded from fitting;
* a 70/15/15 chronological split sensitivity with the selected hyperparameters.

Run: python -m app.ml.robustness --source wheat.csv --runs RF.json XGB.json --output robustness.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import Predictor
from .pipeline import CATEGORIES, _metrics, _read_dataset, _sha256, _split
from .tuning import _estimator, package_versions

ALPHA = 0.10


def _years(frame: pd.DataFrame) -> np.ndarray:
    return frame["campaign"].astype(int).to_numpy()


def extended_metrics(observed, predicted, scale: float | None = None) -> dict:
    """MAE, RMSE, R² and MAPE plus WAPE and, when a naive scale is given, MASE."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    result = _metrics(observed, predicted)
    result["wape"] = float(np.abs(observed - predicted).sum() / np.abs(observed).sum() * 100)
    if scale:
        result["mase"] = float(result["mae"] / scale)
    return result


def block_bootstrap_ci(errors, clusters, seed: int = 42, repetitions: int = 500) -> dict:
    """Percentile 95 % CI of MAE and RMSE resampling whole countries (same scheme as evaluate)."""
    errors, clusters = np.asarray(errors, float), np.asarray(clusters)
    unique = np.unique(clusters)
    counts = np.array([np.sum(clusters == g) for g in unique])
    absolute = np.array([np.abs(errors[clusters == g]).sum() for g in unique])
    squared = np.array([(errors[clusters == g] ** 2).sum() for g in unique])
    samples = np.random.default_rng(seed).integers(0, len(unique), size=(repetitions, len(unique)))
    denominator = counts[samples].sum(axis=1)
    return {"mae": np.quantile(absolute[samples].sum(axis=1) / denominator, [.025, .975]).tolist(),
            "rmse": np.quantile(np.sqrt(squared[samples].sum(axis=1) / denominator), [.025, .975]).tolist(),
            "blocks": int(len(unique)), "repetitions": repetitions}


def persistence(frame: pd.DataFrame, rows: np.ndarray, fallback: float) -> tuple[np.ndarray, int]:
    """Yield of the most recent earlier season of the same region (known before sowing)."""
    years = _years(frame)
    regions = frame["region"].to_numpy()
    target = frame["yield_t_ha"].to_numpy()
    predictions, missing = np.empty(len(rows)), 0
    for position, row in enumerate(rows):
        earlier = np.flatnonzero((regions == regions[row]) & (years < years[row]))
        if len(earlier):
            predictions[position] = target[earlier[np.argmax(years[earlier])]]
        else:
            predictions[position], missing = fallback, missing + 1
    return predictions, missing


def naive_scale(frame: pd.DataFrame, rows: np.ndarray) -> float:
    """In-sample MAE of one-step persistence within each region (MASE denominator)."""
    data = frame.iloc[rows].assign(year=_years(frame)[rows]).sort_values(["region", "year"])
    steps = data.groupby("region")["yield_t_ha"].diff().abs().dropna()
    if steps.empty:
        raise ValueError("MASE needs at least two seasons in one region")
    return float(steps.mean())


def conformal_radius(residuals, alpha: float = ALPHA) -> float | None:
    residuals = np.sort(np.abs(np.asarray(residuals, float)))
    rank = int(np.ceil((len(residuals) + 1) * (1 - alpha)))
    return None if rank > len(residuals) else float(residuals[rank - 1])


def sequential_coverage(years_cal, res_cal, years_test, res_test, mode="expanding",
                        window: int = 4, alpha: float = ALPHA) -> dict:
    """Recalibrate before each test season using only seasons already observed.

    ``static`` keeps the reserved calibration set; ``expanding`` adds every earlier
    test season once its harvest is known; ``sliding`` keeps the last ``window`` seasons.
    The residual of a season is never used to build its own interval.
    """
    years_cal, years_test = np.asarray(years_cal, int), np.asarray(years_test, int)
    res_cal, res_test = np.abs(np.asarray(res_cal, float)), np.abs(np.asarray(res_test, float))
    pool_years = np.concatenate([years_cal, years_test])
    pool_res = np.concatenate([res_cal, res_test])
    by_year, covered = [], np.zeros(len(res_test), bool)
    for year in np.unique(years_test):
        if mode == "static":
            mask = np.isin(pool_years, np.unique(years_cal))
        elif mode == "expanding":
            mask = pool_years < year
        elif mode == "sliding":
            known = np.unique(pool_years[pool_years < year])[-window:]
            mask = np.isin(pool_years, known)
        else:
            raise ValueError("mode must be static, expanding or sliding")
        radius = conformal_radius(pool_res[mask], alpha)
        current = years_test == year
        covered[current] = res_test[current] <= (radius if radius is not None else np.inf)
        by_year.append({"year": int(year), "radius": radius, "calibration_n": int(mask.sum()),
                        "coverage": float(covered[current].mean()), "n": int(current.sum())})
    return {"mode": mode, "nominal": 1 - alpha, "coverage": float(covered.mean()),
            "covered": int(covered.sum()), "n": int(len(covered)), "by_year": by_year,
            "mean_width": float(np.mean([2 * y["radius"] for y in by_year if y["radius"] is not None]))}


def clustered_mean_test(differences, clusters) -> dict:
    """Diebold–Mariano-type test of a mean loss differential with cluster-robust variance.

    Observations in the same country share shocks, so the variance of the mean is
    estimated from country sums (CR1) and referred to a t distribution with G-1 df.
    """
    d = np.asarray(differences, float)
    clusters = np.asarray(clusters)
    groups = np.unique(clusters)
    count = len(groups)
    if count < 3:
        raise ValueError("At least three clusters are required")
    mean = d.mean()
    sums = np.array([np.sum(d[clusters == group] - mean) for group in groups])
    variance = count / (count - 1) * np.sum(sums ** 2) / len(d) ** 2
    statistic = float(mean / np.sqrt(variance)) if variance > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(statistic), df=count - 1))
    return {"mean_difference": float(mean), "statistic": statistic, "df": count - 1, "p_value": p_value}


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_comparison(name_a, errors_a, name_b, errors_b, clusters, seed=42, repetitions=2000) -> dict:
    """Compare two prediction sets on identical rows; positive differences favour B."""
    errors_a, errors_b = np.asarray(errors_a, float), np.asarray(errors_b, float)
    clusters = np.asarray(clusters)
    groups = np.unique(clusters)
    squared = clustered_mean_test(errors_a ** 2 - errors_b ** 2, clusters)
    absolute = clustered_mean_test(np.abs(errors_a) - np.abs(errors_b), clusters)
    country_diff = np.array([np.mean(np.abs(errors_a[clusters == g]) - np.abs(errors_b[clusters == g]))
                             for g in groups])
    nonzero = country_diff[np.abs(country_diff) > 1e-12]
    wilcoxon = stats.wilcoxon(nonzero) if len(nonzero) >= 6 else None
    rng = np.random.default_rng(seed)
    index = [np.flatnonzero(clusters == g) for g in groups]
    deltas = []
    for _ in range(repetitions):
        rows = np.concatenate([index[i] for i in rng.integers(0, len(groups), len(groups))])
        deltas.append(np.sqrt(np.mean(errors_a[rows] ** 2)) - np.sqrt(np.mean(errors_b[rows] ** 2)))
    return {
        "model_a": name_a, "model_b": name_b,
        "rmse_a": float(np.sqrt(np.mean(errors_a ** 2))), "rmse_b": float(np.sqrt(np.mean(errors_b ** 2))),
        "delta_rmse": float(np.sqrt(np.mean(errors_a ** 2)) - np.sqrt(np.mean(errors_b ** 2))),
        "delta_rmse_95ci_cluster_bootstrap": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "dm_squared_loss": squared, "dm_absolute_loss": absolute,
        "wilcoxon_country_mae": None if wilcoxon is None else {
            "statistic": float(wilcoxon.statistic), "p_value": float(wilcoxon.pvalue),
            "countries": int(len(nonzero)),
            "countries_favouring_b": int(np.sum(country_diff > 0))},
    }


def _country_folds(regions: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(regions)
    if len(unique) < folds:
        raise ValueError("Not enough countries for spatial folds")
    shuffled = np.random.default_rng(seed).permutation(unique)
    return [shuffled[k::folds] for k in range(folds)]


def _fit_predict(kind, features, parameters, seed, train: pd.DataFrame, *frames):
    columns = features + list(CATEGORIES)
    estimator = _estimator(kind, features, parameters, seed)
    estimator.fit(train[columns], train.yield_t_ha)
    return [estimator.predict(frame[columns]) for frame in frames]


def spatiotemporal(frame, features, kind, parameters, seed, train, calibration, test, folds=5) -> dict:
    """Fit on training seasons of the other countries; test held-out countries in the test seasons."""
    regions = frame["region"].to_numpy()
    predictions = np.full(len(test), np.nan)
    covered = np.zeros(len(test), bool)
    fold_rows = []
    for number, held_out in enumerate(_country_folds(regions[test], folds, seed)):
        fit_rows = train[~np.isin(regions[train], held_out)]
        cal_rows = calibration[~np.isin(regions[calibration], held_out)]
        target = np.flatnonzero(np.isin(regions[test], held_out))
        cal_pred, test_pred = _fit_predict(kind, features, parameters, seed, frame.iloc[fit_rows],
                                           frame.iloc[cal_rows], frame.iloc[test[target]])
        radius = conformal_radius(frame.iloc[cal_rows].yield_t_ha.to_numpy() - cal_pred)
        predictions[target] = test_pred
        covered[target] = np.abs(frame.iloc[test[target]].yield_t_ha.to_numpy() - test_pred) <= radius
        fold_rows.append({"fold": number, "held_out_countries": int(len(held_out)),
                          "train_rows": int(len(fit_rows)), "calibration_rows": int(len(cal_rows)),
                          "test_rows": int(len(target)), "radius": radius})
    return {"predictions": predictions, "coverage": float(covered.mean()), "folds": fold_rows}


def chronological_split(frame, fractions=(0.70, 0.15)) -> dict:
    """Cut whole seasons so cumulative observations approach 70 % and 85 %."""
    years = _years(frame)
    unique = np.unique(years)
    cumulative = np.cumsum([np.sum(years == y) for y in unique]) / len(years)
    first = unique[np.argmin(np.abs(cumulative - fractions[0]))]
    second = unique[np.argmin(np.abs(cumulative - sum(fractions)))]
    train = np.flatnonzero(years <= first)
    calibration = np.flatnonzero((years > first) & (years <= second))
    test = np.flatnonzero(years > second)
    return {"train": train, "calibration": calibration, "test": test,
            "train_years": [int(unique[0]), int(first)],
            "calibration_years": [int(unique[unique > first][0]), int(second)],
            "test_years": [int(unique[unique > second][0]), int(unique[-1])]}


def run(source, runs, output=None, seed=42, folds=5) -> dict:
    started = time.monotonic()
    frame, features = _read_dataset(source)
    source_hash = _sha256(source)
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in runs]
    for report in reports:
        if report["source_sha256"] != source_hash:
            raise ValueError("A search report was produced from a different dataset")
        if report["validation"] != "temporal":
            raise ValueError("Robustness analyses require temporal searches")
    train, calibration, test, split = _split(frame, "temporal", seed)
    observed = frame.iloc[test].yield_t_ha.to_numpy()
    clusters = frame.iloc[test].region.to_numpy()
    years = _years(frame)
    scale = naive_scale(frame, train)
    train_mean = float(frame.iloc[train].yield_t_ha.mean())
    naive, missing = persistence(frame, test, train_mean)
    errors = {"Persistencia": observed - naive, "Media de entrenamiento": observed - train_mean}
    result = {"source_sha256": source_hash, "seed": seed, "alpha": ALPHA,
              "split": {k: split[k] for k in ("train_years", "validation_years", "test_years",
                                              "train_rows", "validation_rows", "test_rows")},
              "mase_scale_t_ha": scale, "persistence_fallbacks": missing,
              "models": {}, "references": {
                  name: {**extended_metrics(observed, observed - err, scale),
                         "ci95": block_bootstrap_ci(err, clusters, seed)} for name, err in errors.items()}}
    for report in reports:
        kind = report["model_type"]
        predictor = Predictor(report["artifact_path"])
        predicted = predictor(frame.iloc[test])
        cal_res = frame.iloc[calibration].yield_t_ha.to_numpy() - predictor(frame.iloc[calibration])
        errors[kind] = observed - predicted
        parameters = report["best_parameters"]
        st = spatiotemporal(frame, features, kind, parameters, seed, train, calibration, test, folds)
        sensitivity_split = chronological_split(frame)
        cal_pred, test_pred = _fit_predict(kind, features, parameters, seed,
                                           frame.iloc[sensitivity_split["train"]],
                                           frame.iloc[sensitivity_split["calibration"]],
                                           frame.iloc[sensitivity_split["test"]])
        sens_obs = frame.iloc[sensitivity_split["test"]].yield_t_ha.to_numpy()
        sens_radius = conformal_radius(frame.iloc[sensitivity_split["calibration"]].yield_t_ha.to_numpy() - cal_pred)
        residual_years = {int(y): {"mean_residual": float(np.mean(errors[kind][years[test] == y])),
                                   **_metrics(observed[years[test] == y], predicted[years[test] == y])}
                          for y in np.unique(years[test])}
        result["models"][kind] = {
            "parameters": parameters,
            "test": {**extended_metrics(observed, predicted, scale),
                     "ci95": block_bootstrap_ci(errors[kind], clusters, seed)},
            "residuals_by_year": residual_years,
            "calibration_mean_abs_residual": float(np.mean(np.abs(cal_res))),
            "test_mean_abs_residual": float(np.mean(np.abs(errors[kind]))),
            "conformal": {mode: sequential_coverage(years[calibration], cal_res, years[test],
                                                    errors[kind], mode)
                          for mode in ("static", "expanding", "sliding")},
            "unseen_countries": {**extended_metrics(observed, st["predictions"], scale),
                                 "coverage": st["coverage"], "folds": st["folds"]},
            "split_70_15_15": {**{k: sensitivity_split[k] for k in ("train_years", "calibration_years", "test_years")},
                               "rows": [len(sensitivity_split[k]) for k in ("train", "calibration", "test")],
                               "test": extended_metrics(sens_obs, test_pred, scale),
                               "coverage": float(np.mean(np.abs(sens_obs - test_pred) <= sens_radius)),
                               "radius": sens_radius},
        }
    pairs = [(a, b) for a, b in (("RandomForest", "XGBoost"), ("Persistencia", "RandomForest"),
                                 ("Persistencia", "XGBoost")) if a in errors and b in errors]
    comparisons = [paired_comparison(a, errors[a], b, errors[b], clusters, seed) for a, b in pairs]
    adjusted = holm([c["dm_squared_loss"]["p_value"] for c in comparisons])
    for comparison, p in zip(comparisons, adjusted):
        comparison["dm_squared_loss"]["p_holm"] = p
    result["comparisons"] = comparisons
    result["environment"] = {"python": platform.python_version(), "machine": platform.machine(),
                             "processor": platform.processor(), "system": platform.platform(),
                             "versions": package_versions("RandomForest")}
    result["runtime_seconds"] = round(time.monotonic() - started, 2)
    result["notes"] = ["Hyperparameters were selected on the development period only and are reused unchanged.",
                       "Spatio-temporal folds reuse hyperparameters chosen with all development countries.",
                       "Sequential recalibration assumes each harvest is observed before the next season."]
    if output:
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


MODEL_COLORS = {"RandomForest": "#2a78d6", "XGBoost": "#eb6834"}  # validated categorical pair
MODEL_LABELS = {"RandomForest": "Random Forest", "XGBoost": "XGBoost"}
INK, MUTED, GRID, REFERENCE = "#1f1f1f", "#5f5f5f", "#e3e3e0", "#8c8c8c"


def _comma(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def _style(axis, title, ylabel):
    from matplotlib.ticker import FuncFormatter
    axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
    axis.set_ylabel(ylabel, fontsize=9, color=MUTED)
    axis.tick_params(colors=MUTED, labelsize=8.5, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#b5b5b0")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _comma(v, 1)))


def figures(report: dict | str, directory, dpi: int = 600) -> list[str]:
    """Error and coverage figures (PNG, 600 dpi) built only from a robustness report."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    if not isinstance(report, dict):
        report = json.loads(Path(report).read_text(encoding="utf-8"))
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    models = [kind for kind in MODEL_COLORS if kind in report["models"]]
    refs = report["references"]
    paths = []

    # Figure: point error against naive references, and loss of accuracy in unseen countries.
    figure = Figure(figsize=(7.2, 3.3), layout="constrained")
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    names = ["Media de entrenamiento", "Persistencia"] + models
    labels = ["Media", "Persistencia"] + [MODEL_LABELS[m].replace(" ", "\n") for m in models]
    values = [refs[n]["rmse"] for n in names[:2]] + [report["models"][m]["test"]["rmse"] for m in models]
    cis = [refs[n]["ci95"]["rmse"] for n in names[:2]] + [report["models"][m]["test"]["ci95"]["rmse"] for m in models]
    colors = ["#c9c9c4", REFERENCE] + [MODEL_COLORS[m] for m in models]
    x = np.arange(len(values))
    left.bar(x, values, width=0.62, color=colors, zorder=2)
    errors = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(values, cis)]).T
    left.errorbar(x, values, yerr=errors, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
    for xi, (v, (lo, hi)) in enumerate(zip(values, cis)):
        left.text(xi, hi + 0.06, _comma(v, 3), ha="center", va="bottom", fontsize=8, color=INK)
    left.set_xticks(x, labels, fontsize=8.5)
    _style(left, "A  Error en el periodo de prueba", "RMSE (t ha⁻¹)")
    left.set_ylim(0, max(hi for _, hi in cis) * 1.18)
    width = 0.36
    for i, m in enumerate(models):
        seen = report["models"][m]["test"]["rmse"]
        unseen = report["models"][m]["unseen_countries"]["rmse"]
        positions = np.array([0.0, 1.5]) + (i - (len(models) - 1) / 2) * (width + 0.03)
        right.bar(positions, [seen, unseen], width=width, color=MODEL_COLORS[m], label=MODEL_LABELS[m], zorder=2)
        for p, v in zip(positions, [seen, unseen]):
            right.text(p, v + 0.04, _comma(v, 2), ha="center", va="bottom", fontsize=8, color=INK)
    right.axhline(refs["Persistencia"]["rmse"], color=MUTED, linestyle="--", linewidth=1, zorder=1)
    right.text(0.75, refs["Persistencia"]["rmse"] + 0.04, "Persistencia", ha="center", va="bottom",
               fontsize=8, color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    right.set_xticks([0, 1.5], ["Países vistos", "Países no vistos"], fontsize=8.5)
    _style(right, "B  Generalización espacial", "RMSE (t ha⁻¹)")
    right.set_ylim(0, 2.45)
    right.legend(frameon=False, fontsize=8, loc="upper left")
    path = target / "figura_error_referencias.png"
    figure.savefig(path, dpi=dpi, facecolor="white")
    paths.append(str(path))

    # Figure: coverage by season (static) and by recalibration scheme.
    figure = Figure(figsize=(7.2, 3.3), layout="constrained")
    left, right = figure.subplots(1, 2)
    percent = FuncFormatter(lambda v, _: f"{v * 100:.0f} %")
    for m in models:
        rows = report["models"][m]["conformal"]["static"]["by_year"]
        years = [r["year"] for r in rows]
        left.plot(years, [r["coverage"] for r in rows], color=MODEL_COLORS[m], linewidth=2,
                  marker="o", markersize=5, label=MODEL_LABELS[m], zorder=3)
    left.axhline(0.9, color=MUTED, linestyle="--", linewidth=1)
    left.text(years[-1], 0.905, "Objetivo 90 %", ha="right", va="bottom", fontsize=8, color=MUTED)
    _style(left, "A  Cobertura por campaña", "Cobertura empírica")
    left.yaxis.set_major_formatter(percent)
    left.set_ylim(0.6, 1.0)
    left.set_xticks(years, [str(y) for y in years])
    left.legend(frameon=False, fontsize=8, loc="lower left")
    schemes = [("static", "Fija"), ("expanding", "Expansiva"), ("sliding", "Móvil (4)")]
    for i, m in enumerate(models):
        values = [report["models"][m]["conformal"][s]["coverage"] for s, _ in schemes]
        positions = np.arange(len(schemes)) + (i - (len(models) - 1) / 2) * (width + 0.03)
        right.bar(positions, values, width=width, color=MODEL_COLORS[m], label=MODEL_LABELS[m], zorder=2)
        for p, v in zip(positions, values):
            right.text(p, v + 0.008, _comma(v * 100, 1), ha="center", va="bottom", fontsize=7.5, color=INK)
    right.axhline(0.9, color=MUTED, linestyle="--", linewidth=1)
    right.set_xticks(np.arange(len(schemes)), [s for _, s in schemes], fontsize=8.5)
    _style(right, "B  Cobertura por recalibración", "Cobertura empírica")
    right.yaxis.set_major_formatter(percent)
    right.set_ylim(0.6, 1.0)
    path = target / "figura_cobertura.png"
    figure.savefig(path, dpi=dpi, facecolor="white")
    paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="Completed temporal search reports (JSON)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--figures", help="Optional directory for 600 dpi PNG figures")
    args = parser.parse_args()
    report = run(args.source, args.runs, args.output, args.seed, args.folds)
    if args.figures:
        figures(report, args.figures)
    print(json.dumps({k: report[k] for k in ("references", "comparisons")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
