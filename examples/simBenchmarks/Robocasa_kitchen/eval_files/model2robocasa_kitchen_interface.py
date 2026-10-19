"""Policy-side bridge for RoboCasa Kitchen (24-task, PandaOmron) evaluation.

Runs inside the simulator conda environment and talks to the policy server over
a websocket. Everything the model needs to know about the observation contract
(pointmap on/off and frame, state layout and normalization, action chunk) comes
from the server handshake; nothing is inferred from checkpoint files.

Observation conventions of the kitchen fork's ``GrootRoboCasaEnv``:
  * video keys: ``video.res256_image_side_0/1`` + ``video.res256_image_wrist_0``
    (the env renders 512², pads square and resizes to 256² internally)
  * language key: ``annotation.human.action.task_description``
  * state parts: base pose, end-effector pose relative to the base, gripper qpos
"""

from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.rotation_utils import quat_xyzw_to_r6
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# Order MUST match the training-time ``observation.state`` layout:
#   base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4 quat) + gripper_qpos(2) = 16
STATE_KEY_ORDER = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]
_STATE_KEY_DIMS = {
    "state.base_position": 3,
    "state.base_rotation": 4,
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
    "state.gripper_qpos": 2,
}


def _state_slice(key: str) -> slice:
    cursor = 0
    for k in STATE_KEY_ORDER:
        d = _STATE_KEY_DIMS[k]
        if k == key:
            return slice(cursor, cursor + d)
        cursor += d
    raise KeyError(key)


EEF_QUAT_SLICE = _state_slice("state.end_effector_rotation_relative")  # 10:14, xyzw
# The model's state replaces the 4-D end-effector quaternion by its 6-D r6 form.
MODEL_STATE_DIM = sum(_STATE_KEY_DIMS.values()) + 2  # 18

# Camera views fed to the model, in TRAINING order (left, right, wrist).
VIEW_KEY_ORDER = [
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
]

LANG_KEY = "annotation.human.action.task_description"

# Layout of the 12-D environment action returned by the server (r6 already
# decoded to axis-angle server-side); consumed by ``GrootRoboCasaEnv.step``.
ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}
ENV_ACTION_DIM = max(e for (_s, e) in ACTION_SLICES.values())


def noise_seed_for_query(policy_seed: int, episode_seed: int, query_idx: int) -> int:
    """Deterministic per-query seed for the flow-matching sampling noise.

    Distinct policy seeds give independent repetitions of the protocol while the
    environment seeds stay fixed, so re-running a repetition is bit-reproducible.
    """
    return ((int(policy_seed) * 1_000_003 + int(episode_seed)) * 100_003 + int(query_idx)) & 0x7FFFFFFF


class PolicyClient:
    """Single-arm PandaOmron policy client that talks to the websocket server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5678,
        image_size=(224, 224),
        n_action_steps: int = 16,
        policy_seed: int = 0,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.image_size = tuple(image_size)
        self.n_action_steps = int(n_action_steps)
        self.policy_seed = int(policy_seed)
        self.unnorm_key = unnorm_key
        self.task_description = None
        self._episode_seed = 0
        self._query_idx = 0

        meta = self.client.get_server_metadata() or {}
        self.server_meta = meta
        self.action_chunk_size = int(meta["action_chunk_size"])
        if self.n_action_steps > self.action_chunk_size:
            raise RuntimeError(
                f"n_action_steps={self.n_action_steps} exceeds the server's action chunk "
                f"({self.action_chunk_size})")

        # State contract: declared by the server, validated against our layout.
        if meta.get("state_input") != "expert" or meta.get("state_rotation") != "r6":
            raise RuntimeError(
                f"unsupported state contract state_input={meta.get('state_input')!r} "
                f"state_rotation={meta.get('state_rotation')!r}; this client sends the "
                f"{MODEL_STATE_DIM}-D r6 state")
        if int(meta.get("state_dim", -1)) != MODEL_STATE_DIM:
            raise RuntimeError(
                f"server state_dim={meta.get('state_dim')} but the client layout is {MODEL_STATE_DIM}")
        norm = meta.get("state_norm") or {}
        self.state_mean = np.asarray(norm.get("mean", []), dtype=np.float32)
        self.state_std = np.asarray(norm.get("std", []), dtype=np.float32)
        if self.state_mean.shape != (MODEL_STATE_DIM,) or self.state_std.shape != (MODEL_STATE_DIM,):
            raise RuntimeError(
                f"server handshake lacks {MODEL_STATE_DIM}-D state_norm mean/std (got "
                f"{self.state_mean.shape}, {self.state_std.shape})")
        # Constant state dimensions have std 0 in the statistics; training leaves them as-is.
        self.state_std = np.where(self.state_std == 0.0, 1.0, self.state_std)

        # Pointmap contract.
        self.uses_pointmap = bool(meta.get("uses_pointmap"))
        self.pointmap_hw = tuple(int(v) for v in meta.get("pointmap_hw", (256, 256)))
        if self.uses_pointmap and str(meta.get("pointmap_frame")) != "base":
            raise RuntimeError(
                f"checkpoint expects pointmaps in frame {meta.get('pointmap_frame')!r}, but the "
                "evaluation wrapper produces robot-base-frame pointmaps only")
        print(f"[interface] chunk={self.action_chunk_size} execute={self.n_action_steps} "
              f"pointmap={self.uses_pointmap} pointmap_hw={self.pointmap_hw} "
              f"policy_seed={self.policy_seed}")

    # ------------------------------------------------------------------
    def reset(self, task_description) -> None:
        self.task_description = task_description

    def new_episode(self, episode_seed: int) -> None:
        """Restart the per-query noise-seed stream for a new episode."""
        self._episode_seed = int(episode_seed)
        self._query_idx = 0

    def step(self, observations: Dict) -> Dict:
        """One policy query for a batch of environments.

        ``observations`` is the dict produced by ``MultiStepWrapper`` through a
        gymnasium vector env: videos are (B, 1, H, W, 3), states (B, 1, d) and the
        pointmap, if enabled, (B, 1, V, h, w, 3) with its (B, 1, V, h, w) mask.
        """
        task_descs = observations[LANG_KEY]
        instructions = list(task_descs) if isinstance(task_descs, (tuple, list)) else [task_descs]
        if instructions[0] != self.task_description:
            self.reset(instructions[0])

        views = [observations[k] for k in VIEW_KEY_ORDER]  # each (B, 1, H, W, 3)
        batch_size = len(views[0])
        images = [[self._resize_image(v[b][-1]) for v in views] for b in range(batch_size)]

        raw_state = np.concatenate([observations[k] for k in STATE_KEY_ORDER], axis=-1)  # (B, 1, 16)
        model_state = self._encode_state(raw_state)  # (B, 1, 18)

        examples = []
        for b in range(batch_size):
            example = {
                "image": images[b],
                "lang": instructions[b] if b < len(instructions) else instructions[0],
                "state": model_state[b],
            }
            if self.uses_pointmap:
                assert "state.pointmap" in observations, (
                    "the checkpoint uses pointmaps but the observation has no state.pointmap; "
                    "PointmapObservationWrapper must be enabled")
                example["pointmap"] = np.asarray(observations["state.pointmap"][b][-1], dtype=np.float16)
                example["pointmap_valid"] = np.asarray(observations["state.pointmap_valid"][b][-1], dtype=bool)
            examples.append(example)

        vla_input = {
            "examples": examples,
            "unnorm_key": self.unnorm_key,
            "noise_seed": noise_seed_for_query(self.policy_seed, self._episode_seed, self._query_idx),
        }
        self._query_idx += 1
        response = self.client.predict_action(vla_input)
        raw_actions = np.array(response["data"]["actions"])  # (B, chunk, 12), env units
        assert raw_actions.shape[-1] == ENV_ACTION_DIM, (
            f"server returned action dim {raw_actions.shape[-1]}, expected {ENV_ACTION_DIM}")

        out = {}
        for key, (s, e) in ACTION_SLICES.items():
            out[key] = raw_actions[:, : self.n_action_steps, s:e]
        return {"actions": out}

    # ------------------------------------------------------------------
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def _encode_state(self, raw_state: np.ndarray) -> np.ndarray:
        """16-D simulator state -> 18-D r6 state -> mean/std normalized (float32)."""
        q = raw_state[..., EEF_QUAT_SLICE]  # xyzw, as robosuite emits
        state = np.concatenate([
            raw_state[..., :EEF_QUAT_SLICE.start],
            quat_xyzw_to_r6(q).astype(raw_state.dtype),
            raw_state[..., EEF_QUAT_SLICE.stop:],
        ], axis=-1)
        assert state.shape[-1] == MODEL_STATE_DIM
        return ((state - self.state_mean) / self.state_std).astype(np.float32)
