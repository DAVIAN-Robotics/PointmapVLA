"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}


# ---------------------------------------------------------------------------
# r6 LIBERO (end-effector rotation stored as the first two rows of the rotation
# matrix) and its pointmap pairing. Dataset roots:
#   data_root_dir      -> the r6 RGB/state/action dataset (4 suites)
#   pointmap_data_root -> the paired depth + camera-feature dataset (same 4 suite names)
# ---------------------------------------------------------------------------
class LiberoR6DataConfig:
    """r6 LIBERO: state 11 (mean_std), action 10 (min_max), horizon 8.

    Group keys differ from ``Libero4in1DataConfig`` (``state.roll`` etc.), so
    ``data_root_dir`` must point at the r6 dataset.
    """

    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
    ]
    # Per-key widths of the model-facing vectors (same order as the key lists above).
    state_key_dims = {
        "state.end_effector_position_relative": 3,
        "state.end_effector_rotation_relative": 6,
        "state.gripper_qpos": 2,
    }
    action_key_dims = {
        "action.end_effector_position": 3,
        "action.end_effector_rotation": 6,
        "action.gripper_close": 1,
    }
    STATE_DIM = 11
    ACTION_DIM = 10
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    @classmethod
    def validate_key_dims(cls):
        assert sum(cls.state_key_dims.values()) == cls.STATE_DIM
        assert sum(cls.action_key_dims.values()) == cls.ACTION_DIM
        assert list(cls.state_key_dims) == cls.state_keys, "state key order drift"
        assert list(cls.action_key_dims) == cls.action_keys, "action key order drift"

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys + self.action_keys),
            StateActionTransform(
                apply_to=self.state_keys + self.action_keys,
                normalization_modes={
                    **{k: "mean_std" for k in self.state_keys},
                    **{k: "min_max" for k in self.action_keys},
                },
            ),
        ])


LiberoR6DataConfig.validate_key_dims()


class LiberoR6PointmapDataConfig(LiberoR6DataConfig):
    """The r6 recipe plus a ROBOT-BASE-frame pointmap per camera view.

    Pointmap dataset contents per suite:
      - ``depth.{agentview,wrist}``: FFV1 lossless gray16le, uint16 = meters * 10000
      - ``{view}_camera_intrinsics`` (4) / ``_camera_extrinsics`` (7, pos + quat wxyz, cam->world)
      - ``state.base_position`` (3) / ``state.base_rotation`` (4, xyzw)

    The intrinsics carry a NEGATIVE fx: the RGB the model sees is the 180-degree
    rotated observation, which is the mirror image of the geometrically correct view,
    and a mirror cannot be expressed as a camera rotation. Encoding it as fx' = -fx,
    cx' = W - cx keeps the back-projection arithmetic unchanged while aligning the
    pointmap pixel-for-pixel with the image.

    ``PointmapDerive`` runs FIRST so ``frame='base'`` sees the RAW base pose before
    normalization. The 11-D r6 state carries no base pose, so the pointmap dataset
    supplies ``state.base_position`` / ``state.base_rotation`` (``pm_extra_state_keys``).
    """

    pm_views = {
        "agentview": {"depth": "depth.agentview",
                      "intrinsics": "state.agentview_camera_intrinsics",
                      "extrinsics": "state.agentview_camera_extrinsics"},
        "wrist": {"depth": "depth.wrist",
                  "intrinsics": "state.wrist_camera_intrinsics",
                  "extrinsics": "state.wrist_camera_extrinsics"},
    }

    def make_dataset(self, dataset_path, modality_configs, transforms, embodiment_tag,
                     video_backend, delete_pause_frame, data_cfg, dataset_name):
        """``lerobot_datasets.make_LeRobotSingleDataset`` opt-in hook: build a
        :class:`PointmapPairedDataset` whose secondary is ``<pointmap_data_root>/<suite>``."""
        import os

        from starVLA.dataloader.gr00t_lerobot.paired_pointmap_dataset import PointmapPairedDataset

        pm_root = (data_cfg or {}).get("pointmap_data_root")
        assert pm_root, (
            "pointmap mixture needs datasets.vla_data.pointmap_data_root in the yaml "
            "(the root holding the four paired LIBERO suites)"
        )
        pm_path = os.path.join(pm_root, dataset_name) if dataset_name else pm_root
        return PointmapPairedDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            pm_dataset_path=pm_path,
            pm_views=self.pm_views,
            pm_out_hw=tuple((data_cfg or {}).get("pointmap_out_hw", (128, 128))),
            pm_extra_state_keys=("state.base_position", "state.base_rotation"),
        )

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.pointmap import PointmapDerive

        parent = super().transform()
        return ComposedModalityTransform(
            transforms=[PointmapDerive(views=self.pm_views, frame="base"), *parent.transforms]
        )


LiberoR6PointmapDataConfig.validate_key_dims()

ROBOT_TYPE_CONFIG_MAP["libero_franka_r6"] = LiberoR6DataConfig()
ROBOT_TYPE_CONFIG_MAP["libero_franka_r6_pointmap"] = LiberoR6PointmapDataConfig()

_LIBERO_SUITES = [
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]
DATASET_NAMED_MIXTURES["libero_r6"] = [(s, 1.0, "libero_franka_r6") for s in _LIBERO_SUITES]
DATASET_NAMED_MIXTURES["libero_r6_pointmap"] = [(s, 1.0, "libero_franka_r6_pointmap") for s in _LIBERO_SUITES]
