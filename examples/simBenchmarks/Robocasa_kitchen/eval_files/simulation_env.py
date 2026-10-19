"""RoboCasa Kitchen (24-task, PandaOmron) evaluation client for a PointmapVLA
websocket policy server.

Protocol (per task): 50 episodes x 720 steps, action chunk 16 executed in full,
success = any substep of a chunk reports success (episode ends there).
Episode i is seeded with ``seed * 100000 + i`` so the evaluated episode set is
identical for any number of parallel environments.

Run from the repository root inside the simulator conda environment:

    python -m examples.simBenchmarks.Robocasa_kitchen.eval_files.simulation_env \\
        --args.port 5678 --args.env-name OpenSingleDoor --args.n-episodes 50 \\
        --args.result-dir results/robocasa/rep0/eval_json --args.policy-seed 0
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import numpy as np

import tyro

# Importing this module registers the kitchen envs ``robocasa_panda_omron/<Task>_PandaOmron_Env``.
importlib.import_module("robocasa.utils.gym_utils.gymnasium_groot")

from examples.simBenchmarks.Robocasa_kitchen.eval_files.model2robocasa_kitchen_interface import PolicyClient
from examples.simBenchmarks.Robocasa_kitchen.eval_files.multistep_wrapper import MultiStepWrapper

ENV_ID_PREFIX = "robocasa_panda_omron/"
ENV_ID_SUFFIX = "_PandaOmron_Env"

# Shared numpy pointmap math, loaded by file path: the simulator environment
# cannot import the starVLA package, and the training loader uses this same file.
# eval_files -> Robocasa_kitchen -> simBenchmarks -> examples -> <repo root>
_POINTMAP_MATH = Path(__file__).resolve().parents[4] / "starVLA/dataloader/gr00t_lerobot/transform/pointmap_math.py"


def load_pointmap_math():
    spec = importlib.util.spec_from_file_location("pointmap_math_eval", _POINTMAP_MATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_to_env_id(name: str) -> str:
    """Accept either a bare task name (``OpenSingleDoor``) or a full gym id."""
    if name.startswith(ENV_ID_PREFIX):
        return name
    return f"{ENV_ID_PREFIX}{name}{ENV_ID_SUFFIX}"


def env_id_to_task(env_id: str) -> str:
    name = env_id[len(ENV_ID_PREFIX):] if env_id.startswith(ENV_ID_PREFIX) else env_id
    return name[: -len(ENV_ID_SUFFIX)] if name.endswith(ENV_ID_SUFFIX) else name


class EpisodeSeedWrapper(gym.Wrapper):
    """Deterministic per-episode seeding: ``seed = base_seed * 100000 + episode_id``.

    Env ``i`` of ``K`` parallel envs owns global episodes ``i, i+K, i+2K, ...`` so the
    evaluated episode set does not depend on ``K``. The kitchen env samples layout,
    style and object placement from its own ``rng`` attribute, which the gymnasium
    seed path never touches, so that generator is re-created here as well.
    """

    def __init__(self, env: gym.Env, base_seed: int, episode_start: int = 0, episode_step: int = 1,
                 seed_stride: int = 100000):
        super().__init__(env)
        self._base_seed = base_seed
        self._seed_stride = seed_stride
        self._episode_id = episode_start
        self._episode_step = episode_step

    def reset(self, seed=None, options=None):
        seed = self._base_seed * self._seed_stride + self._episode_id
        rs_env = getattr(self.env.unwrapped, "env", None)
        if rs_env is not None and hasattr(rs_env, "rng"):
            rs_env.rng = np.random.default_rng(seed)
        self._episode_id += self._episode_step
        return super().reset(seed=seed, options=options)


class PointmapObservationWrapper(gym.Wrapper):
    """Render depth online and emit robot-base-frame pointmaps for the three views.

    Per camera: 256² depth render, linearized with ``get_real_depth_map``, metric
    range 0.05 < z < 6.5 m, per-step intrinsics/extrinsics (the wrist camera moves),
    back-projection with the same ``derive_pointmap`` the training loader uses, then
    ``world_to_base`` with the base pose from the env's own observation. Invalid
    pixels are zero. Output ``state.pointmap`` (V, h, w, 3) float16 and
    ``state.pointmap_valid`` (V, h, w) bool.

    The ``state.`` prefix is deliberate: ``MultiStepWrapper`` only stacks keys that
    start with ``video``/``state``.
    """

    CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

    def __init__(self, env: gym.Env, out_hw=(256, 256), render_res: int = 256, z_range=(0.05, 6.5)):
        super().__init__(env)
        self._res = int(render_res)
        self._out_hw = tuple(int(v) for v in out_hw)
        self._z_range = z_range
        assert self._res % self._out_hw[0] == 0, f"render_res {self._res} must be a multiple of out_hw {self._out_hw}"
        self._pm_math = load_pointmap_math()
        V, (h, w) = len(self.CAMERAS), self._out_hw
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            space = dict(self.env.observation_space.spaces)
            space["state.pointmap"] = gym.spaces.Box(-np.inf, np.inf, (V, h, w, 3), np.float16)
            space["state.pointmap_valid"] = gym.spaces.Box(0, 1, (V, h, w), np.bool_)
            self.observation_space = gym.spaces.Dict(space)

    @staticmethod
    def _quat_wxyz_from_mat(R: np.ndarray) -> np.ndarray:
        import robosuite.utils.transform_utils as T  # returns xyzw

        q = T.mat2quat(np.asarray(R, dtype=np.float64))
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

    def _render_depth(self, sim, cam: str) -> np.ndarray:
        """Depth buffer in [0, 1]. EGL occasionally returns an out-of-range buffer;
        re-render up to twice, then clamp so the episode can continue."""
        for attempt in range(3):
            _rgb, dbuf = sim.render(width=self._res, height=self._res, camera_name=cam, depth=True)
            dbuf = np.asarray(dbuf)
            if np.all(np.isfinite(dbuf)) and dbuf.min() >= 0.0 and dbuf.max() <= 1.0:
                return dbuf
            if attempt == 2:
                bad = float(np.mean(~np.isfinite(dbuf) | (dbuf < 0.0) | (dbuf > 1.0)))
                print(f"[pointmap-wrapper] depth glitch persists on {cam}; clamping (bad={bad:.1%})", flush=True)
        return np.clip(np.nan_to_num(dbuf, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    def _augment(self, obs: dict) -> dict:
        import robosuite.utils.camera_utils as CU

        sim = self.env.unwrapped.env.sim
        base_pos = np.asarray(obs["state.base_position"], dtype=np.float64).reshape(-1)[:3]
        base_rot = np.asarray(obs["state.base_rotation"], dtype=np.float64).reshape(-1)[:4]  # xyzw
        sh = self._res // self._out_hw[0]
        pms, pvs = [], []
        for cam in self.CAMERAS:
            dbuf = self._render_depth(sim, cam)
            z = np.asarray(CU.get_real_depth_map(sim, dbuf[::-1]), dtype=np.float64)
            z = z[..., 0] if z.ndim == 3 else z
            z = np.where((z > self._z_range[0]) & (z < self._z_range[1]), z, 0.0)
            K = CU.get_camera_intrinsic_matrix(sim, cam, self._res, self._res)
            E = CU.get_camera_extrinsic_matrix(sim, cam)  # 4x4 cam->world, OpenCV axes
            extr7 = np.concatenate([E[:3, 3], self._quat_wxyz_from_mat(E[:3, :3])])
            pm, valid = self._pm_math.derive_pointmap(z, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], extr7)
            pm = self._pm_math.world_to_base(pm, base_pos, base_rot)
            pm = np.where(valid[..., None], pm, 0.0).astype(np.float32)
            pms.append(pm[::sh, ::sh, :])
            pvs.append(valid[::sh, ::sh])
        obs = dict(obs)
        obs["state.pointmap"] = np.stack(pms).astype(np.float16)
        obs["state.pointmap_valid"] = np.stack(pvs)
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info


@dataclass
class SimulationConfig:
    env_name: str
    n_episodes: int = 50
    n_envs: int = 1
    seed: int = 42
    pointmap: bool = True
    pointmap_hw: Tuple[int, int] = (256, 256)
    n_action_steps: int = 16
    max_episode_steps: int = 720


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    env = gym.make(config.env_name, enable_render=True)
    # Innermost so the injected seed reaches the robosuite env's reset.
    env = EpisodeSeedWrapper(env, base_seed=config.seed, episode_start=idx, episode_step=config.n_envs)
    if config.pointmap:
        env = PointmapObservationWrapper(env, out_hw=config.pointmap_hw)
    env = MultiStepWrapper(
        env,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=config.n_action_steps,
        max_episode_steps=config.max_episode_steps,
        terminate_on_success=True,
    )
    return env


def run_simulation(client: PolicyClient, config: SimulationConfig) -> Tuple[str, List[bool]]:
    print(f"[robocasa_kitchen] running {config.n_episodes} eps on {config.env_name}")
    env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
    env = (
        gym.vector.SyncVectorEnv(env_fns)
        if config.n_envs == 1
        else gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    )

    # Env i owns global episodes {i, i+K, ...} and completes exactly its quota, so
    # the scored set is global episodes 0..n_episodes-1 for any K.
    K = config.n_envs
    quotas = [len(range(i, config.n_episodes, K)) for i in range(K)]
    done_counts = [0] * K
    local_counts = [0] * K
    ep_results: dict = {}
    current_successes = [False] * K
    obs, _ = env.reset()
    # Sampling-noise seeding. K == 1: one noise stream per episode, restarted at
    # each episode boundary. K > 1: one request carries the whole batch, so a single
    # run-level stream is used; reruns with the same K are still bit-reproducible.
    client.new_episode(config.seed * 100000)
    t0 = time.time()
    while sum(done_counts) < config.n_episodes:
        actions = client.step(obs)["actions"]
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        for i in range(K):
            # success = any substep of the chunk. On the terminating step gymnasium
            # moves the final info into infos["final_info"][i]; read both.
            if "success" in env_infos:
                current_successes[i] |= bool(np.any(env_infos["success"][i]))
            final_infos = env_infos.get("final_info")
            if final_infos is not None and final_infos[i] is not None and "success" in final_infos[i]:
                current_successes[i] |= bool(np.any(final_infos[i]["success"]))
            if terminations[i] or truncations[i]:
                gid = i + local_counts[i] * K
                local_counts[i] += 1
                if done_counts[i] < quotas[i]:
                    ep_results[gid] = current_successes[i]
                    done_counts[i] += 1
                current_successes[i] = False
                if K == 1:
                    client.new_episode(config.seed * 100000 + local_counts[i])
        obs = next_obs

    episode_successes = [bool(ep_results[g]) for g in sorted(ep_results)]
    try:
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"[robocasa_kitchen] env.close ignored: {e}")
    print(f"[robocasa_kitchen] {len(episode_successes)} eps done in {time.time() - t0:.1f}s")
    return config.env_name, episode_successes


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    env_name: str = "OpenSingleDoor"
    n_episodes: int = 50
    n_envs: int = 1
    max_episode_steps: int = 720
    n_action_steps: int = 16
    resize_size: Tuple[int, int] = (224, 224)
    # Environment seed; episode i uses seed * 100000 + i.
    seed: int = 42
    # Seed of the policy's sampling noise; repetitions of the protocol use 0, 1, 2.
    policy_seed: int = 0
    # Where <Task>.json goes.
    result_dir: str = "results/robocasa/eval_json"
    # Dataset-statistics key for un-normalization; None = the checkpoint's single key.
    unnorm_key: Optional[str] = None


def main(args: Args) -> None:
    logging.info(json.dumps(dataclasses.asdict(args), indent=2, default=str))
    client = PolicyClient(
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        policy_seed=args.policy_seed,
        unnorm_key=args.unnorm_key,
    )
    cfg = SimulationConfig(
        env_name=task_to_env_id(args.env_name),
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        seed=args.seed,
        pointmap=client.uses_pointmap,
        pointmap_hw=client.pointmap_hw,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )
    env_id, successes = run_simulation(client, cfg)
    sr = float(np.mean(successes)) if successes else 0.0
    print(f"\n=== {env_id} ===")
    print(f"success rate: {sr:.2f} ({sum(successes)}/{len(successes)})")
    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env_id_to_task(env_id)}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump({"env": env_id, "success_rate": sr, "successes": successes,
                   "seed": args.seed, "policy_seed": args.policy_seed, "n_envs": args.n_envs}, f, indent=2)
    tmp_path.replace(out_path)  # atomic publish: an existing JSON means "done"


if __name__ == "__main__":
    tyro.cli(main)
