# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.

PointmapVLA checkpoints additionally declare an "expert" state input at the
handshake (`state_input="expert"`, `state_rotation="r6"`, `state_dim=11`,
`state_norm`). For those, the example must carry the raw proprioception parts
(`eef_pos` (3,), `eef_quat` (4,) xyzw, `gripper_qpos` (2,)); this client turns
them into the normalized 11-D r6 state the model was trained on. If the server
declares `uses_pointmap`, the example must also carry `pointmap` /
`pointmap_valid` (see `libero_pointmap.py`).
"""

from collections import deque
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.rotation_utils import quat_xyzw_to_r6
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

EXPERT_STATE_DIM = 11  # eef position 3 + r6 rotation 6 + gripper qpos 2


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size: Sequence[int] = (224, 224),
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self._server_metadata = meta

        self.image_size: tuple = tuple(image_size)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"server_meta: {meta} ***"
        )

        # Observation contract declared by the server; the client never guesses.
        # A pointmap checkpoint evaluated without pointmaps degrades silently.
        self._uses_pointmap = bool(meta.get("uses_pointmap"))
        self._expert_state = meta.get("state_input") == "expert"
        self._state_mean = self._state_std = None
        if self._expert_state:
            if str(meta.get("state_rotation")) != "r6":
                raise RuntimeError(
                    f"server state_rotation={meta.get('state_rotation')!r}, expected 'r6'")
            if int(meta.get("state_dim") or 0) != EXPERT_STATE_DIM:
                raise RuntimeError(
                    f"server state_dim={meta.get('state_dim')}, client layout implies "
                    f"{EXPERT_STATE_DIM} (pos3 + r6 + gripper2)")
            norm = meta.get("state_norm") or {}
            self._state_mean = np.asarray(norm.get("mean", []), dtype=np.float64)
            self._state_std = np.maximum(np.asarray(norm.get("std", []), dtype=np.float64), 1e-8)
            if self._state_mean.shape != (EXPERT_STATE_DIM,) or self._state_std.shape != (EXPERT_STATE_DIM,):
                raise RuntimeError(
                    f"server handshake lacks {EXPERT_STATE_DIM}-D state_norm mean/std")
            print(f"*** expert state ON: r6 / mean_std over {EXPERT_STATE_DIM} dims ***")

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            from examples.simBenchmarks.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler

            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None

    def _build_payload(self, example: dict) -> dict:
        """One observation -> server payload (image resize + state r6 conversion
        and normalization). Shared by ``step`` and ``chunk_batch`` so both runners
        send exactly the same quantity."""
        if self.image_size and example.get("image"):
            resized = []
            target_hw = tuple(self.image_size)  # (H, W)
            for img in example["image"]:
                arr = np.asarray(img)
                if tuple(arr.shape[:2]) != target_hw:
                    arr = np.asarray(
                        Image.fromarray(arr).resize(
                            (target_hw[1], target_hw[0]), Image.BILINEAR
                        )
                    )
                resized.append(arr)
            example = {**example, "image": resized}
        if self._uses_pointmap and "pointmap" not in example:
            raise RuntimeError(
                "the server declared uses_pointmap but the example has no 'pointmap'; "
                "use eval_libero_vec.py, which renders pointmaps online")
        payload = dict(example)
        if self._expert_state:
            for k in ("eef_pos", "eef_quat", "gripper_qpos"):
                if k not in example:
                    raise RuntimeError(f"expert-state checkpoint but the example lacks {k!r}")
            raw = np.concatenate([
                np.asarray(example["eef_pos"], dtype=np.float64).reshape(3),
                quat_xyzw_to_r6(np.asarray(example["eef_quat"], dtype=np.float64).reshape(4)),
                np.asarray(example["gripper_qpos"], dtype=np.float64).reshape(2),
            ])  # (11,), same layout as the training dataset
            norm = (raw - self._state_mean) / self._state_std
            payload["state"] = norm[None, :].astype(np.float32)  # (frames=1, D)
        for k in ("eef_pos", "eef_quat", "gripper_qpos"):
            payload.pop(k, None)
        return payload

    def _vla_input(self, payloads: list) -> dict:
        return {
            "examples": payloads,
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

    def chunk_batch(self, examples: list) -> np.ndarray:
        """B observations -> (B, T, D) action chunks in env units (no chunk cache;
        the vectorized runner owns per-slot scheduling)."""
        payloads = [self._build_payload(e) for e in examples]
        response = self.client.predict_action(self._vla_input(payloads))
        try:
            actions = np.asarray(response["data"]["actions"])  # (B, T, D)
        except KeyError:
            raise KeyError(f"'actions' missing from response: keys={list(response.get('data', {}).keys())}")
        assert actions.shape[0] == len(examples), (
            f"server returned {actions.shape[0]} chunks for a batch of {len(examples)}")
        return actions

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Refresh chunk if needed.
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            vla_input = self._vla_input([self._build_payload(example)])
            # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
            # Embodied policies degrade SILENTLY (no error) when the eval-time observation
            # differs from what the model saw during TRAINING. Verify these match the
            # training config used for this checkpoint:
            #   - state       : whether proprioceptive state is included (and its dim/order/normalization))
            #   - image size  : resize / crop resolution (e.g. 224x224)
            #   - image count : how many camera views are fed
            #   - image order : the ordering of those camera views
            #   - action normalization: unnorm_key must match the training dataset stats
            # ==============================================================================
            response = self.client.predict_action(vla_input)
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            self.raw_actions = np.asarray(actions_batch)[0]  # (T, D)

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {"raw_action": raw_action}

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
