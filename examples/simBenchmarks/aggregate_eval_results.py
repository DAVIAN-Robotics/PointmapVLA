#!/usr/bin/env python
"""Aggregate evaluation results with integrity checks.

One CLI, three benchmarks. Each positional directory is one independent run
(a RoboCasa repetition or a LIBERO seed); the summary reports mean +- sd
(sample sd, ddof=1) over runs.

    aggregate_eval_results.py robocasa    results/robocasa/rep0 results/robocasa/rep1 ...
    aggregate_eval_results.py libero      results/libero/s7 results/libero/s8 ...
    aggregate_eval_results.py libero_plus results/libero_plus/s7 ...

Integrity comes before numbers: a run that does not cover the full protocol
(24 tasks x 50 episodes; 4 suites x 10 tasks tiled by shards without gaps or
overlaps; LIBERO-Plus totals 2402/2518/2591/2519) is reported as INCOMPLETE and
no rate is printed unless ``--allow-partial`` is given. A missing shard would
otherwise shift the success rate silently and plausibly.

Directory layouts accepted per run: the JSON files directly in the directory, or
in an ``eval_json/`` (RoboCasa), ``libero/`` or ``libero_plus/`` subdirectory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ROBOCASA_TASKS = [
    "PnPMicrowaveToCounter", "PnPCounterToMicrowave", "TurnOffStove", "CoffeeSetupMug",
    "PnPCabToCounter", "PnPCounterToCab", "PnPCounterToSink", "PnPCounterToStove",
    "PnPStoveToCounter", "PnPSinkToCounter", "OpenDoubleDoor", "TurnOnMicrowave",
    "CoffeeServeMug", "OpenSingleDoor", "TurnOnStove", "TurnSinkSpout",
    "OpenDrawer", "TurnOffSinkFaucet", "CoffeePressButton", "CloseDoubleDoor",
    "TurnOnSinkFaucet", "TurnOffMicrowave", "CloseSingleDoor", "CloseDrawer",
]
ROBOCASA_EPISODES = 50
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LIBERO_TASKS = {s: 10 for s in LIBERO_SUITES}
LIBERO_TRIALS = 50
LIBERO_PLUS_TASKS = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}
LIBERO_PLUS_TRIALS = 1


class Incomplete(Exception):
    """A run does not cover the full protocol."""


def _json_files(run_dir: str, pattern: str, subdirs) -> list:
    for sub in ("",) + tuple(subdirs):
        files = sorted(glob.glob(os.path.join(run_dir, sub, pattern)))
        if files:
            return files
    raise Incomplete(f"{run_dir}: no {pattern} files found")


def _env_to_task(env: str) -> str:
    name = env.split("/")[-1]
    return name[: -len("_PandaOmron_Env")] if name.endswith("_PandaOmron_Env") else name


def _mean_sd(values):
    values = [float(v) for v in values]
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return mean, sd


def _fmt(mean, sd):
    return f"{100 * mean:.2f}" + (f" ± {100 * sd:.2f}" if sd is not None else "")


# ------------------------------------------------------------------ RoboCasa
def load_robocasa_run(run_dir: str, episodes: int = ROBOCASA_EPISODES) -> dict:
    """-> {task: success_rate} for exactly the 24 tasks x `episodes` episodes."""
    per_task, problems = {}, []
    for f in _json_files(run_dir, "*.json", ("eval_json",)):
        d = json.load(open(f))
        task = _env_to_task(d["env"])
        succ = d["successes"]
        if task in per_task:
            problems.append(f"duplicate result for {task}")
        if len(succ) != episodes:
            problems.append(f"{task}: {len(succ)} episodes, expected {episodes}")
        per_task[task] = sum(bool(s) for s in succ) / max(len(succ), 1)
    missing = sorted(set(ROBOCASA_TASKS) - set(per_task))
    extra = sorted(set(per_task) - set(ROBOCASA_TASKS))
    if missing:
        problems.append(f"missing tasks: {missing}")
    if extra:
        problems.append(f"unknown tasks: {extra}")
    if problems:
        raise Incomplete(f"{run_dir}: " + "; ".join(problems))
    return per_task


def aggregate_robocasa(run_dirs, episodes=ROBOCASA_EPISODES, allow_partial=False):
    runs = []
    for d in run_dirs:
        try:
            runs.append(load_robocasa_run(d, episodes))
        except Incomplete as e:
            if not allow_partial:
                raise
            print(f"WARNING (partial): {e}", file=sys.stderr)
            if "no *.json" in str(e):
                continue
            runs.append(load_robocasa_run_partial(d))
    if not runs:
        raise Incomplete("no usable runs")
    tasks = sorted({t for r in runs for t in r}, key=lambda t: ROBOCASA_TASKS.index(t) if t in ROBOCASA_TASKS else 99)
    table = {}
    for t in tasks:
        vals = [r[t] for r in runs if t in r]
        table[t] = {"mean": _mean_sd(vals)[0], "sd": _mean_sd(vals)[1], "per_run": vals}
    run_scores = [statistics.mean(r.values()) for r in runs]
    mean, sd = _mean_sd(run_scores)
    lines = ["| Task | " + " | ".join(f"run{i}" for i in range(len(runs))) + " | mean ± sd |",
             "|---|" + "---|" * len(runs) + "---|"]
    for t in tasks:
        row = table[t]
        cells = " | ".join(f"{100 * v:.1f}" for v in row["per_run"])
        lines.append(f"| {t} | {cells} | {_fmt(row['mean'], row['sd'])} |")
    lines.append("| **Average** | " + " | ".join(f"{100 * s:.2f}" for s in run_scores) + f" | **{_fmt(mean, sd)}** |")
    summary = {"benchmark": "robocasa", "runs": list(run_dirs), "episodes_per_task": episodes,
               "per_task": table, "run_scores": run_scores, "mean": mean, "sd": sd}
    return "\n".join(lines), summary


def load_robocasa_run_partial(run_dir: str) -> dict:
    per_task = {}
    for f in _json_files(run_dir, "*.json", ("eval_json",)):
        d = json.load(open(f))
        succ = d["successes"]
        per_task[_env_to_task(d["env"])] = sum(bool(s) for s in succ) / max(len(succ), 1)
    return per_task


# ------------------------------------------------------------------ LIBERO (shared shard logic)
def _load_shards(run_dir: str, subdir: str) -> list:
    return [json.load(open(f)) for f in _json_files(run_dir, "*_shard*.json", (subdir,))]


def _check_shards(run_dir, shards, expect_tasks, trials, require_categories):
    """Integrity: shards tile [0, n) per suite without gaps/overlaps (I1), every
    task id appears exactly once with `trials` episodes (I2), per-suite totals
    equal n_tasks x trials (I3), category counts add up to the total."""
    problems = []
    spans = defaultdict(list)
    seen = defaultdict(list)
    totals = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    for s in shards:
        suite = s["task_suite"]
        if suite not in expect_tasks:
            problems.append(f"unknown suite {suite!r}")
            continue
        spans[suite].append(tuple(s["shard"]))
        totals[suite][0] += s["total_episodes"]
        totals[suite][1] += s["total_successes"]
        for tid, rec in s["per_task"].items():
            seen[suite].append(int(tid))
            if rec["episodes"] != trials:
                problems.append(f"{suite} task {tid}: {rec['episodes']} episodes, expected {trials}")
        for cat, v in (s.get("by_category") or {}).items():
            by_cat[cat][0] += v["total_count"]
            by_cat[cat][1] += v["success_count"]
    for suite, n in expect_tasks.items():
        cur, gaps, overlaps = 0, [], []
        for lo, hi in sorted(spans.get(suite, [])):
            if lo > cur:
                gaps.append((cur, lo))
            elif lo < cur:
                overlaps.append((lo, min(hi, cur)))
            cur = max(cur, hi)
        if cur < n:
            gaps.append((cur, n))
        if gaps or overlaps:
            problems.append(f"{suite}: shard gaps {gaps} overlaps {overlaps}")
        ids = seen.get(suite, [])
        if sorted(set(ids)) != list(range(n)) or len(ids) != n:
            problems.append(f"{suite}: task ids do not cover [0,{n}) exactly once ({len(set(ids))} unique, {len(ids)} total)")
        if totals[suite][0] != n * trials:
            problems.append(f"{suite}: {totals[suite][0]} episodes, expected {n * trials}")
    if require_categories:
        cat_total = sum(v[0] for v in by_cat.values())
        all_total = sum(v[0] for v in totals.values())
        if cat_total != all_total:
            problems.append(f"category counts {cat_total} != episodes {all_total}")
    if problems:
        raise Incomplete(f"{run_dir}: " + "; ".join(problems))
    return totals, by_cat


def _load_libero_like(run_dir, subdir, expect_tasks, trials, require_categories, allow_partial):
    shards = _load_shards(run_dir, subdir)
    try:
        return _check_shards(run_dir, shards, expect_tasks, trials, require_categories)
    except Incomplete as e:
        if not allow_partial:
            raise
        print(f"WARNING (partial): {e}", file=sys.stderr)
        totals = defaultdict(lambda: [0, 0])
        by_cat = defaultdict(lambda: [0, 0])
        for s in shards:
            totals[s["task_suite"]][0] += s["total_episodes"]
            totals[s["task_suite"]][1] += s["total_successes"]
            for cat, v in (s.get("by_category") or {}).items():
                by_cat[cat][0] += v["total_count"]
                by_cat[cat][1] += v["success_count"]
        return totals, by_cat


def _rate(pair):
    return pair[1] / pair[0] if pair[0] else float("nan")


def aggregate_libero(run_dirs, trials=LIBERO_TRIALS, allow_partial=False):
    runs = [_load_libero_like(d, "libero", LIBERO_TASKS, trials, False, allow_partial)[0] for d in run_dirs]
    return _libero_table("libero", run_dirs, runs, None, trials)


def aggregate_libero_plus(run_dirs, trials=LIBERO_PLUS_TRIALS, allow_partial=False):
    loaded = [_load_libero_like(d, "libero_plus", LIBERO_PLUS_TASKS, trials, True, allow_partial) for d in run_dirs]
    return _libero_table("libero_plus", run_dirs, [t for t, _ in loaded], [c for _, c in loaded], trials)


def _libero_table(bench, run_dirs, run_totals, run_cats, trials):
    suites = [s for s in LIBERO_SUITES if any(s in t for t in run_totals)]
    lines = ["| | " + " | ".join(f"run{i}" for i in range(len(run_totals))) + " | mean ± sd |",
             "|---|" + "---|" * len(run_totals) + "---|"]
    summary = {"benchmark": bench, "runs": list(run_dirs), "trials_per_task": trials, "per_suite": {}}

    def add_rows(label_to_pairs, section):
        for label, pairs in label_to_pairs:
            rates = [_rate(p) for p in pairs]
            mean, sd = _mean_sd(rates)
            section[label] = {"mean": mean, "sd": sd, "per_run": rates,
                              "episodes_per_run": [p[0] for p in pairs]}
            lines.append(f"| {label} | " + " | ".join(f"{100 * r:.2f}" for r in rates) + f" | {_fmt(mean, sd)} |")

    add_rows([(s, [t.get(s, [0, 0]) for t in run_totals]) for s in suites], summary["per_suite"])
    overall_pairs = [[sum(v[0] for v in t.values()), sum(v[1] for v in t.values())] for t in run_totals]
    overall = [_rate(p) for p in overall_pairs]
    mean, sd = _mean_sd(overall)
    lines.append("| **Overall** | " + " | ".join(f"{100 * r:.2f}" for r in overall) + f" | **{_fmt(mean, sd)}** |")
    summary.update({"run_scores": overall, "mean": mean, "sd": sd})
    if run_cats is not None:
        cats = sorted({c for rc in run_cats for c in rc})
        summary["by_category"] = {}
        lines.append("| | " + " | ".join("" for _ in run_totals) + " | |")
        add_rows([(c, [rc.get(c, [0, 0]) for rc in run_cats]) for c in cats], summary["by_category"])
    return "\n".join(lines), summary


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["robocasa", "libero", "libero_plus"])
    ap.add_argument("run_dirs", nargs="+", help="one directory per repetition / seed")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes per task (RoboCasa) or trials per task (LIBERO); default = protocol value")
    ap.add_argument("--out", default=None,
                    help="where to write the JSON summary (default: summary.json in the run directory, "
                         "or in the common parent of several run directories)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="print rates even when a run does not cover the full protocol (never for reporting)")
    a = ap.parse_args(argv)
    try:
        if a.benchmark == "robocasa":
            table, summary = aggregate_robocasa(a.run_dirs, a.episodes or ROBOCASA_EPISODES, a.allow_partial)
        elif a.benchmark == "libero":
            table, summary = aggregate_libero(a.run_dirs, a.episodes or LIBERO_TRIALS, a.allow_partial)
        else:
            table, summary = aggregate_libero_plus(a.run_dirs, a.episodes or LIBERO_PLUS_TRIALS, a.allow_partial)
    except Incomplete as e:
        print(f"INCOMPLETE: {e}\nNo rates printed; pass --allow-partial to see partial numbers.", file=sys.stderr)
        return 1
    print(table)
    summary["partial"] = bool(a.allow_partial)
    if a.out is None:
        base = a.run_dirs[0] if len(a.run_dirs) == 1 else os.path.commonpath([os.path.abspath(d) for d in a.run_dirs])
        a.out = os.path.join(base, "summary.json")
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
