"""Unit test for examples/simBenchmarks/aggregate_eval_results.py on synthetic
results: exact hand-computed numbers, and refusal on partial coverage.

Run with pytest, or directly: python tests/eval/test_aggregate.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_AGG = Path(__file__).resolve().parents[2] / "examples/simBenchmarks/aggregate_eval_results.py"
spec = importlib.util.spec_from_file_location("aggregate_eval_results", _AGG)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


# ------------------------------------------------------------------ RoboCasa
def _robocasa_rep(root, rep, successes_by_task):
    for task, succ in successes_by_task.items():
        _write(root / f"rep{rep}" / "eval_json" / f"{task}.json",
               {"env": f"robocasa_panda_omron/{task}_PandaOmron_Env",
                "success_rate": sum(succ) / len(succ), "successes": succ})


def test_robocasa_mean_sd(tmp_path):
    # 4 episodes per task; rep0 all tasks 2/4, rep1 all tasks 4/4 except one task 0/4.
    rep0 = {t: [True, True, False, False] for t in agg.ROBOCASA_TASKS}
    rep1 = {t: [True] * 4 for t in agg.ROBOCASA_TASKS}
    rep1["OpenDrawer"] = [False] * 4
    _robocasa_rep(tmp_path, 0, rep0)
    _robocasa_rep(tmp_path, 1, rep1)
    table, s = agg.aggregate_robocasa([str(tmp_path / "rep0"), str(tmp_path / "rep1")], episodes=4)
    assert s["run_scores"][0] == pytest.approx(0.5)
    assert s["run_scores"][1] == pytest.approx(23 / 24)
    assert s["mean"] == pytest.approx((0.5 + 23 / 24) / 2)
    # sample sd of two values = |a-b| / sqrt(2)
    assert s["sd"] == pytest.approx(abs(0.5 - 23 / 24) / 2 ** 0.5)
    assert s["per_task"]["OpenDrawer"]["mean"] == pytest.approx(0.25)
    assert s["per_task"]["CloseDrawer"]["mean"] == pytest.approx(0.75)
    assert "| **Average** | 50.00 | 95.83 |" in table


def test_robocasa_refuses_missing_task(tmp_path):
    rep0 = {t: [True] * 4 for t in agg.ROBOCASA_TASKS[:-1]}  # 23 of 24 tasks
    _robocasa_rep(tmp_path, 0, rep0)
    with pytest.raises(agg.Incomplete, match="missing tasks"):
        agg.aggregate_robocasa([str(tmp_path / "rep0")], episodes=4)
    # wrong episode count is also refused
    rep0 = {t: [True] * 3 for t in agg.ROBOCASA_TASKS}
    _robocasa_rep(tmp_path / "short", 0, rep0)
    with pytest.raises(agg.Incomplete, match="expected 4"):
        agg.aggregate_robocasa([str(tmp_path / "short" / "rep0")], episodes=4)


# ------------------------------------------------------------------ LIBERO
def _shard(suite, lo, hi, trials, succ_per_task, category=None):
    per_task = {}
    by_cat = {}
    for tid in range(lo, hi):
        rec = {"name": f"{suite}_task{tid}", "episodes": trials, "successes": succ_per_task}
        if category:
            rec["category"] = category
            d = by_cat.setdefault(category, {"total_count": 0, "success_count": 0})
            d["total_count"] += trials
            d["success_count"] += succ_per_task
        per_task[str(tid)] = rec
    out = {"task_suite": suite, "shard": [lo, hi], "num_trials_per_task": trials,
           "total_episodes": trials * (hi - lo), "total_successes": succ_per_task * (hi - lo),
           "per_task": per_task, "runner": "vec", "k": 2}
    if category:
        out["by_category"] = by_cat
    return out


def _libero_seed(root, seed, trials, succ_by_suite):
    for suite in agg.LIBERO_SUITES:
        for lo, hi in ((0, 5), (5, 10)):
            _write(root / f"s{seed}" / "libero" / f"{suite}_shard{lo:05d}-{hi:05d}.json",
                   _shard(suite, lo, hi, trials, succ_by_suite[suite]))


def test_libero_exact(tmp_path):
    # 2 trials per task; seed7: every suite 2/2 except libero_10 1/2; seed8: all 1/2.
    _libero_seed(tmp_path, 7, 2, {"libero_spatial": 2, "libero_object": 2, "libero_goal": 2, "libero_10": 1})
    _libero_seed(tmp_path, 8, 2, {s: 1 for s in agg.LIBERO_SUITES})
    table, s = agg.aggregate_libero([str(tmp_path / "s7"), str(tmp_path / "s8")], trials=2)
    assert s["run_scores"][0] == pytest.approx(70 / 80)
    assert s["run_scores"][1] == pytest.approx(0.5)
    assert s["per_suite"]["libero_10"]["mean"] == pytest.approx(0.5)
    assert s["per_suite"]["libero_spatial"]["per_run"] == [1.0, 0.5]
    assert s["mean"] == pytest.approx((0.875 + 0.5) / 2)
    assert s["sd"] == pytest.approx(abs(0.875 - 0.5) / 2 ** 0.5)
    assert "| **Overall** | 87.50 | 50.00 |" in table


def test_libero_refuses_missing_and_overlapping_shards(tmp_path):
    _libero_seed(tmp_path, 7, 2, {s: 1 for s in agg.LIBERO_SUITES})
    os.remove(tmp_path / "s7" / "libero" / "libero_goal_shard00005-00010.json")
    with pytest.raises(agg.Incomplete, match=r"libero_goal: shard gaps \[\(5, 10\)\]"):
        agg.aggregate_libero([str(tmp_path / "s7")], trials=2)
    # --allow-partial still produces numbers
    _, s = agg.aggregate_libero([str(tmp_path / "s7")], trials=2, allow_partial=True)
    assert s["per_suite"]["libero_goal"]["episodes_per_run"] == [10]

    _libero_seed(tmp_path / "dup", 7, 2, {s: 1 for s in agg.LIBERO_SUITES})
    _write(tmp_path / "dup" / "s7" / "libero" / "libero_10_shard00003-00007.json",
           _shard("libero_10", 3, 7, 2, 1))
    with pytest.raises(agg.Incomplete, match="overlaps"):
        agg.aggregate_libero([str(tmp_path / "dup" / "s7")], trials=2)


# ------------------------------------------------------------------ LIBERO-Plus
def test_libero_plus_categories_and_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "LIBERO_PLUS_TASKS", {s: 4 for s in agg.LIBERO_SUITES})
    root = tmp_path / "s7" / "libero_plus"
    for suite in agg.LIBERO_SUITES:
        _write(root / f"{suite}_shard00000-00002.json", _shard(suite, 0, 2, 1, 1, "Camera Viewpoints"))
        _write(root / f"{suite}_shard00002-00004.json", _shard(suite, 2, 4, 1, 0, "Light Conditions"))
    table, s = agg.aggregate_libero_plus([str(tmp_path / "s7")], trials=1)
    assert s["run_scores"] == [pytest.approx(0.5)]
    assert s["sd"] is None
    assert s["by_category"]["Camera Viewpoints"]["mean"] == pytest.approx(1.0)
    assert s["by_category"]["Light Conditions"]["mean"] == pytest.approx(0.0)
    assert s["by_category"]["Camera Viewpoints"]["episodes_per_run"] == [8]
    assert "| Light Conditions | 0.00 | 0.00 |" in table

    # I3: a suite with the wrong total (expected 4 tasks, one shard missing) is refused
    os.remove(root / "libero_10_shard00002-00004.json")
    with pytest.raises(agg.Incomplete, match="libero_10"):
        agg.aggregate_libero_plus([str(tmp_path / "s7")], trials=1)


def test_cli_exit_code_on_partial(tmp_path):
    _libero_seed(tmp_path, 7, 2, {s: 1 for s in agg.LIBERO_SUITES})
    os.remove(tmp_path / "s7" / "libero" / "libero_goal_shard00005-00010.json")
    out = tmp_path / "summary.json"
    assert agg.main(["libero", str(tmp_path / "s7"), "--episodes", "2", "--out", str(out)]) == 1
    assert not out.exists()
    assert agg.main(["libero", str(tmp_path / "s7"), "--episodes", "2", "--out", str(out), "--allow-partial"]) == 0
    assert json.loads(out.read_text())["partial"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
