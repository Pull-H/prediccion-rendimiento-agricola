import numpy as np
import pandas as pd
import pytest

from app.ml.robustness import (chronological_split, clustered_mean_test, conformal_radius,
                               extended_metrics, holm, naive_scale, paired_comparison,
                               persistence, sequential_coverage, _country_folds)


def _panel():
    rows = [{"region": f"C{c}", "campaign": str(year), "yield_t_ha": c + 0.1 * (year - 2000)}
            for c in range(6) for year in range(2000, 2010) if not (c == 0 and year == 2003)]
    return pd.DataFrame(rows)


def test_persistence_uses_only_earlier_seasons_of_the_same_region():
    frame = _panel()
    rows = np.flatnonzero(frame.campaign.astype(int) == 2004)
    predicted, missing = persistence(frame, rows, fallback=-1.0)
    assert missing == 0
    for row, value in zip(rows, predicted):
        region = frame.region.iloc[row]
        # Region C0 lacks 2003, so its last known season is 2002.
        expected_year = 2002 if region == "C0" else 2003
        expected = frame[(frame.region == region) & (frame.campaign == str(expected_year))].yield_t_ha.item()
        assert value == pytest.approx(expected)
    first = np.flatnonzero(frame.campaign.astype(int) == 2000)
    assert persistence(frame, first, fallback=-1.0)[1] == len(first)


def test_extended_metrics_and_naive_scale():
    frame = _panel()
    scale = naive_scale(frame, np.arange(len(frame)))
    assert scale > 0
    metrics = extended_metrics([2.0, 4.0], [1.0, 5.0], scale=scale)
    assert metrics["wape"] == pytest.approx(100 * 2 / 6)
    assert metrics["mase"] == pytest.approx(1.0 / scale)


def test_sequential_conformal_never_uses_the_current_or_future_season():
    years_cal = np.repeat([2005, 2006], 20)
    years_test = np.repeat([2007, 2008, 2009], 20)
    res_cal = np.full(len(years_cal), 0.1)
    # A huge residual in 2009 must not widen the intervals of 2007 or 2008.
    res_test = np.concatenate([np.full(20, 0.5), np.full(20, 0.5), np.full(20, 50.0)])
    static = sequential_coverage(years_cal, res_cal, years_test, res_test, "static")
    expanding = sequential_coverage(years_cal, res_cal, years_test, res_test, "expanding")
    assert all(entry["radius"] == pytest.approx(0.1) for entry in static["by_year"])
    by_year = {entry["year"]: entry for entry in expanding["by_year"]}
    assert by_year[2007]["calibration_n"] == 40 and by_year[2007]["radius"] == pytest.approx(0.1)
    assert by_year[2009]["calibration_n"] == 80 and by_year[2009]["radius"] < 50
    sliding = sequential_coverage(years_cal, res_cal, years_test, res_test, "sliding", window=2)
    assert {entry["year"]: entry["calibration_n"] for entry in sliding["by_year"]}[2009] == 40
    assert conformal_radius(np.ones(5), alpha=0.1) is None


def test_clustered_test_and_holm_adjustment():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(30), 5)
    null = clustered_mean_test(rng.normal(0, 1, len(clusters)), clusters)
    shifted = clustered_mean_test(rng.normal(2, 1, len(clusters)), clusters)
    assert 0 <= null["p_value"] <= 1 and shifted["p_value"] < 0.001
    assert null["df"] == 29
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    comparison = paired_comparison("A", rng.normal(0, 2, 150), "B", rng.normal(0, 1, 150),
                                   clusters, repetitions=200)
    low, high = comparison["delta_rmse_95ci_cluster_bootstrap"]
    assert comparison["delta_rmse"] > 0 and low < high
    assert comparison["wilcoxon_country_mae"]["countries"] == 30


def test_spatial_folds_and_chronological_split_are_disjoint():
    regions = np.repeat([f"C{i}" for i in range(12)], 3)
    folds = _country_folds(regions, 5, seed=1)
    flattened = np.concatenate(folds)
    assert sorted(flattened) == sorted(np.unique(regions)) and len(flattened) == 12
    split = chronological_split(_panel())
    years = _panel().campaign.astype(int).to_numpy()
    assert years[split["train"]].max() < years[split["calibration"]].min()
    assert years[split["calibration"]].max() < years[split["test"]].min()
