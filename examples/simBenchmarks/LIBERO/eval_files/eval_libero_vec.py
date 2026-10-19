"""Vectorized LIBERO / LIBERO-Plus evaluation runner (K environment slots, one
batched policy query per action chunk).

Why batched: a single policy query dominates the serial runner's per-step cost,
while the server's latency grows slowly with batch size. K slots advancing
together give a several-fold speedup at identical success rates.

Structure (one round = one action chunk):
  1. collect the observations of all live slots -> ONE batched query -> (B, chunk, 7)
  2. every slot executes its chunk locally in a subprocess (stops early on done)
  3. finished slots write their result and take the next episode

Reproducibility: the batch composition is deterministic for a given K (fixed work
order, aligned round boundaries). The same K and seed reproduce the same result;
a different K changes which noise draw each episode receives, so keep K fixed
within a set of results that are compared to each other.

Output: one shard JSON per (suite, task range), see ``write_result``.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys
import time
from collections import OrderedDict, deque

import numpy as np

# eval_files -> LIBERO -> simBenchmarks -> examples -> <repo root>
REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[4])

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Same values as the serial runner (eval_libero.py); the protocol must not diverge.
MAX_STEPS = {
    "libero_spatial": 220,   # longest training demo 193
    "libero_object": 280,    # 254
    "libero_goal": 300,      # 270
    "libero_10": 520,        # 505
    "libero_90": 400,        # 373
}


def _binarize_gripper_open(open_val) -> float:
    """Same rule as eval_libero.py: open > 0.5 -> -1 (open), else +1 (close)."""
    v = float(np.asarray(open_val, dtype=np.float32).reshape(-1)[0])
    return 1.0 - 2.0 * (v > 0.5)


# ------------------------------------------------------------------ env subprocess
def _find_noise_env(env):
    """LIBERO-Plus wraps the env with a sensor-noise layer that owns a ``noise``
    attribute; vanilla LIBERO has none (returns None)."""
    base = env
    for _ in range(6):
        if base is None:
            return None
        if hasattr(base, "noise"):
            return base
        base = getattr(base, "env", None)
    return None


def _step_chunk(env, actions, budget, noise_env, skip_noise, stop_on_done=True):
    """Execute a chunk; return (last obs, done, steps used).

    skip_noise: LIBERO-Plus applies its sensor-noise image corruption on every
    ``env.step``. Only the last observation of a chunk reaches the policy, so the
    corruption of intermediate steps is computed for nobody; skipping it makes
    that axis several times faster with the same policy inputs. The corruption
    functions draw from np.random, so skipping changes the realized noise sample
    (not its kind or strength); ``--keep-unused-noise`` restores the exact
    upstream sequence.
    """
    n = min(len(actions), budget)
    done, used, obs = False, 0, None
    nv = getattr(noise_env, "noise", 0) if (skip_noise and noise_env is not None) else 0
    for j in range(n):
        if nv:
            noise_env.noise = 0 if j < n - 1 else nv
        obs, _, done, _ = env.step(list(map(float, actions[j])))
        used += 1
        if done and stop_on_done:
            break
    if nv:
        noise_env.noise = nv
    return obs, done, used


def _env_worker(conn, resolution: int, seed: int, skip_noise: bool = True,
                want_pm: bool = False, pm_out_hw=(256, 256)):
    """Owns one slot. Commands: next / run / close."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    LP = None
    if want_pm:
        sys.path.insert(0, REPO_ROOT)
        from examples.simBenchmarks.LIBERO.eval_files import libero_pointmap as LP

    env = None
    cur_bddl = None
    noise_env = None
    sim = None
    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "close":
                break
            if cmd == "next":
                # make (if the task changed) + reset + set_init_state + settle, in one round trip.
                bddl, init_state, num_wait = arg
                if bddl != cur_bddl:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                             camera_heights=resolution,
                                             camera_widths=resolution,
                                             camera_depths=bool(want_pm))
                    env.seed(seed)
                    cur_bddl = bddl
                    noise_env = _find_noise_env(env)
                env.reset()
                obs = env.set_init_state(init_state)
                sim = env.env.sim  # reset re-creates MjSim; grab it afterwards
                # The settling steps never stop on done (same as the serial runner).
                obs, _, _ = _step_chunk(env, [LIBERO_DUMMY_ACTION] * num_wait,
                                        num_wait, noise_env, skip_noise, stop_on_done=False)
                conn.send(("obs", _pack_obs(obs, sim, LP, pm_out_hw, resolution)))
            elif cmd == "run":
                actions, budget = arg  # (T, 7) float, remaining step budget
                obs, done, used = _step_chunk(env, actions, budget, noise_env, skip_noise)
                conn.send(("ran", (_pack_obs(obs, sim, LP, pm_out_hw, resolution), bool(done), int(used))))
            else:
                conn.send(("err", f"unknown cmd {cmd}"))
    except Exception:
        import traceback
        try:
            conn.send(("fatal", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


def _pack_obs(obs, sim=None, LP=None, pm_out_hw=(256, 256), res: int = 256):
    """Observation sent once per chunk: rot180 RGB (dataset convention), raw
    proprioception parts, and the base-frame pointmap when requested."""
    if obs is None:
        return None
    out = {
        "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
        "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64),  # xyzw
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
    }
    if LP is not None:
        pm, pv = LP.derive_from_obs(sim, obs, res=res, out_hw=tuple(pm_out_hw))
        out["pointmap"], out["pointmap_valid"] = pm, pv
    return out


class Slot:
    """One subprocess plus the state of the episode it is currently running."""

    def __init__(self, idx: int, resolution: int, seed: int, ctx, skip_noise: bool = True,
                 want_pm: bool = False, pm_out_hw=(256, 256)):
        self.idx = idx
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_env_worker,
                                args=(child, resolution, seed, skip_noise, want_pm, pm_out_hw),
                                daemon=True)
        self.proc.start()
        child.close()
        self.item = None          # (task_id, ep_idx)
        self.obs = None
        self.steps = 0
        self.done = False
        self.refilling = False    # "next" sent, observation not yet received

    def send(self, cmd, arg=None):
        self.conn.send((cmd, arg))

    def recv(self):
        kind, payload = self.conn.recv()
        if kind in ("err", "fatal"):
            raise RuntimeError(f"slot{self.idx} failed:\n{payload}")
        return payload

    def ready(self) -> bool:
        return self.conn.poll()

    def close(self):
        try:
            self.send("close")
            self.proc.join(timeout=20)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()


# ------------------------------------------------------------------ main
def load_task_categories(bench: str, suite_name: str) -> dict:
    """LIBERO-Plus: task id (1-based) -> (perturbation category, name)."""
    if bench != "libero_plus":
        return {}
    home = os.environ.get("LIBERO_HOME", "")
    path = os.path.join(home, "libero/libero/benchmark/task_classification.json")
    with open(path) as f:
        return {item["id"]: (item["category"], item["name"]) for item in json.load(f)[suite_name]}


def write_result(log_path, bench, suite_name, t_start, t_end, trials, per_task, name_of, id2cat, K):
    total_ep = sum(v["episodes"] for v in per_task.values())
    total_su = sum(v["successes"] for v in per_task.values())
    per_task_res, by_cat = {}, {}
    for tid, v in per_task.items():
        rec = {"name": name_of[tid], "episodes": v["episodes"], "successes": v["successes"]}
        if bench == "libero_plus":
            cat = id2cat.get(tid + 1, ("unknown", ""))[0]
            rec["category"] = cat
            d = by_cat.setdefault(cat, {"total_count": 0, "success_count": 0})
            d["total_count"] += v["episodes"]
            d["success_count"] += v["successes"]
        per_task_res[int(tid)] = rec
    out = {"task_suite": suite_name, "shard": [int(t_start), int(t_end)],
           "num_trials_per_task": int(trials),
           "total_episodes": int(total_ep), "total_successes": int(total_su),
           "per_task": per_task_res, "runner": "vec", "k": int(K)}
    if bench == "libero_plus":
        out["by_category"] = by_cat
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    name = f"{suite_name}_shard{t_start:05d}-{t_end:05d}.json"
    final = os.path.join(log_path, name)
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, final)  # atomic publish: an existing JSON means "done"
    return name, total_ep, total_su


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["libero", "libero_plus"], default="libero")
    ap.add_argument("--task-suite-name", required=True)
    ap.add_argument("--task-id-start", type=int, default=0)
    ap.add_argument("--task-id-end", type=int, default=-1)
    ap.add_argument("--num-trials-per-task", type=int, default=50)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="policy server port (required unless --pm-selftest)")
    ap.add_argument("--log-path", required=True, help="directory for the shard JSON")
    ap.add_argument("--seed", type=int, default=7, help="environment seed")
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--k", type=int, default=12, help="parallel env slots = policy batch size")
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--pm-out-hw", type=int, default=None,
                    help="pointmap resolution; default = the server's declared pointmap_hw")
    ap.add_argument("--pm-selftest", action="store_true",
                    help="no policy server: check that a slot renders and ships base-frame pointmaps, then exit")
    ap.add_argument("--keep-unused-noise", action="store_true",
                    help="LIBERO-Plus: also apply sensor-noise corruption on intermediate steps (slower, "
                         "bit-exact with the upstream sequence)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    from libero.libero import benchmark, get_libero_path
    sys.path.insert(0, REPO_ROOT)
    from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient

    suite = benchmark.get_benchmark_dict()[a.task_suite_name]()
    n_tasks = suite.n_tasks
    t_start = a.task_id_start
    t_end = n_tasks if a.task_id_end < 0 else min(a.task_id_end, n_tasks)
    max_steps = MAX_STEPS[a.task_suite_name] + a.num_steps_wait
    logging.info(f"[vec] suite={a.task_suite_name} shard=[{t_start},{t_end}) of {n_tasks} "
                 f"EPS={a.num_trials_per_task} K={a.k} max_steps={max_steps}")
    id2cat = load_task_categories(a.bench, a.task_suite_name)

    def bddl_path(tid):
        task = suite.get_task(tid)
        return str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)

    if a.pm_selftest:
        hw = a.pm_out_hw or 256
        ctx0 = mp.get_context("spawn")
        sl = Slot(0, LIBERO_ENV_RESOLUTION, a.seed, ctx0, True, True, (hw, hw))
        sl.send("next", (bddl_path(t_start), suite.get_task_init_states(t_start)[0], a.num_steps_wait))
        o = sl.recv()
        pm, pv = np.asarray(o["pointmap"], np.float32), np.asarray(o["pointmap_valid"])
        v = pm[pv]
        print(f"[selftest] pointmap {pm.shape} {pm.dtype} valid {100 * pv.mean():.1f}% "
              f"x[{v[:, 0].min():+.2f},{v[:, 0].max():+.2f}] z[{v[:, 2].min():+.2f},{v[:, 2].max():+.2f}] m")
        assert pm.shape == (2, hw, hw, 3), f"shape {pm.shape} violates the (2, {hw}, {hw}, 3) contract"
        assert pv.mean() > 0.5, "less than half of the pixels are valid; check camera_depths"
        assert 0.3 < float(v.max() - v.min()) < 20.0, "values are not in metres"
        sl.close()
        print("[selftest] passed: slots produce base-frame pointmaps")
        return

    if not a.port:
        ap.error("--port is required")
    client = ModelClient(host=a.host, port=a.port, unnorm_key=a.unnorm_key)
    chunk = client.action_chunk_size
    meta = client._server_metadata or {}
    want_pm = bool(meta.get("uses_pointmap"))
    pm_hw = (a.pm_out_hw, a.pm_out_hw) if a.pm_out_hw else tuple(int(x) for x in meta.get("pointmap_hw", (256, 256)))
    if want_pm:
        frame = meta.get("pointmap_frame")
        assert frame in (None, "base"), f"server wants pointmap_frame={frame!r}; only 'base' is produced"
        logging.info(f"[vec] pointmap ON (frame={frame!r}, out_hw={pm_hw})")

    bddl_of = {tid: bddl_path(tid) for tid in range(t_start, t_end)}
    inits_of = {tid: suite.get_task_init_states(tid) for tid in range(t_start, t_end)}
    name_of = {tid: str(suite.get_task(tid).name if a.bench == "libero"
                        else id2cat.get(tid + 1, ("", suite.get_task(tid).name))[1])
               for tid in range(t_start, t_end)}
    langs = {tid: suite.get_task(tid).language for tid in range(t_start, t_end)}

    per_task = {tid: {"episodes": 0, "successes": 0} for tid in range(t_start, t_end)}
    ctx = mp.get_context("spawn")  # fork would inherit a broken EGL context

    # Task-major work list: a slot keeps picking episodes of its current task, so
    # the env is reused instead of re-created at every episode.
    pending = OrderedDict((tid, deque(range(a.num_trials_per_task))) for tid in range(t_start, t_end))
    n_work = sum(len(d) for d in pending.values())
    K = max(1, min(a.k, n_work))
    slots = [Slot(i, LIBERO_ENV_RESOLUTION, a.seed, ctx, not a.keep_unused_noise, want_pm, pm_hw)
             for i in range(K)]
    logging.info(f"[vec] {K} slots, {n_work} episodes")

    def take(sl) -> bool:
        tid = sl.item[0] if sl.item else None
        if tid is not None and pending.get(tid):
            sl.item = (tid, pending[tid].popleft())
            return True
        for t2, dq in pending.items():
            if dq:
                sl.item = (t2, dq.popleft())
                return True
        sl.item = None
        return False

    def send_next(sl):
        """Asynchronous reset so a refill never stalls the round."""
        tid, ep = sl.item
        # LIBERO ships 50 init states per task; more trials cycle through them.
        n_init = len(inits_of[tid])
        sl.send("next", (bddl_of[tid], inits_of[tid][ep % n_init], a.num_steps_wait))
        sl.refilling = True

    for sl in slots:
        if take(sl):
            send_next(sl)

    t0 = time.time()
    n_ep_done = 0
    live = []

    while True:
        # (a) slots whose refill finished join the round (non-blocking poll)
        for sl in slots:
            if sl.refilling and sl.ready():
                sl.obs = sl.recv(); sl.steps = a.num_steps_wait
                sl.done = False; sl.refilling = False
                live.append(sl)
        if not live:
            waiting = [sl for sl in slots if sl.refilling]
            if not waiting:
                break  # no work left and nothing refilling
            sl = waiting[0]
            sl.obs = sl.recv(); sl.steps = a.num_steps_wait
            sl.done = False; sl.refilling = False
            live.append(sl)

        # (b) exactly one batched query per round
        examples = []
        for sl in live:
            ex = {"image": [sl.obs["image"], sl.obs["wrist"]],
                  "lang": langs[sl.item[0]],
                  "eef_pos": sl.obs["eef_pos"], "eef_quat": sl.obs["eef_quat"],
                  "gripper_qpos": sl.obs["gripper_qpos"]}
            if want_pm:
                ex["pointmap"] = np.asarray(sl.obs["pointmap"], dtype=np.float16)
                ex["pointmap_valid"] = np.asarray(sl.obs["pointmap_valid"], dtype=bool)
            examples.append(ex)
        acts = client.chunk_batch(examples)  # (B, chunk, D)

        # (c) execute chunks in the slots (parallel). Cast each piece to float32
        # like the serial runner does; float64 can flip borderline episodes.
        for sl, ac in zip(live, acts):
            env_acts = np.stack([
                np.concatenate([
                    np.asarray(ac[j, :3], dtype=np.float32).reshape(-1),
                    np.asarray(ac[j, 3:6], dtype=np.float32).reshape(-1),
                    np.asarray([_binarize_gripper_open(ac[j, 6:7])], dtype=np.float32),
                ], axis=0)
                for j in range(min(chunk, ac.shape[0]))
            ])
            sl.send("run", (env_acts, max_steps - sl.steps))
        for sl in live:
            obs, done, used = sl.recv()
            if obs is not None:
                sl.obs = obs
            sl.steps += used
            sl.done = done

        # (d) finished slots record their result and take the next episode
        still = []
        any_fin = False
        for sl in live:
            if sl.done or sl.steps >= max_steps:
                tid = sl.item[0]
                per_task[tid]["episodes"] += 1
                per_task[tid]["successes"] += int(sl.done)
                n_ep_done += 1; any_fin = True
                if take(sl):
                    send_next(sl)
                else:
                    sl.close()
            else:
                still.append(sl)
        live = still
        if any_fin:
            el = time.time() - t0
            logging.info(f"[vec] {n_ep_done}/{n_work} ep, {el:.0f}s, {el / max(n_ep_done, 1):.2f} s/ep, "
                         f"SR {sum(v['successes'] for v in per_task.values())}/{n_ep_done}")

    for sl in slots:
        sl.close()

    name, total_ep, total_su = write_result(a.log_path, a.bench, a.task_suite_name, t_start, t_end,
                                            a.num_trials_per_task, per_task, name_of, id2cat, K)
    el = time.time() - t0
    logging.info(f"[vec] done SR {total_su}/{total_ep} = {total_su / max(total_ep, 1):.3f}, "
                 f"{el:.0f}s ({el / max(total_ep, 1):.2f} s/ep) -> {name}")


if __name__ == "__main__":
    main()
