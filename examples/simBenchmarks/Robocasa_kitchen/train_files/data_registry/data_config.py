"""RoboCasa Kitchen 24-task (PandaOmron) -- data configs and mixtures.

Loaded automatically by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``.

Datasets (LeRobot v3.0, MimicGen 300 demos/task, 7,200 episodes):

* ``robocasa-MG_300`` (RGB / state / action, the "r6" bake):
  - observation.state (18-D): eef_pos(3) + eef_rot_r6(6) + gripper_qpos(2) + base_pos(3) + base_quat(4)
  - action (15-D): eef_pos(3) + eef_rot_r6(6) + gripper_close(1) + base_motion(4) + control_mode(1)
  - 3 cameras at 128x128, 20 fps
* ``robocasa-MG_300-pointmap`` (episode-1:1 with the above): per-view lossless depth
  (FFV1 gray16le, uint16 = meters * 10000, 0 invalid) plus per-frame camera intrinsics
  (fx, fy, cx, cy) and extrinsics (pos + quat wxyz, cam->world, OpenCV).

"r6" means the end-effector rotation is stored as the first two rows of the rotation
matrix (6 numbers) instead of a quaternion / axis-angle. The model-facing state and
action vectors are the concatenation of ``state_keys`` / ``action_keys`` in the order
given below; the evaluation client builds its state vector in the same order.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboCasaKitchenR6DataConfig:
    """RGB-only recipe: state 18 (mean_std), action 15 (min_max), horizon 16."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    state_keys = [
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    # Per-key widths of the model-facing vectors (same order as the key lists above).
    state_key_dims = {
        "state.base_position": 3,
        "state.base_rotation": 4,
        "state.end_effector_position_relative": 3,
        "state.end_effector_rotation_relative": 6,
        "state.gripper_qpos": 2,
    }
    action_key_dims = {
        "action.end_effector_position": 3,
        "action.end_effector_rotation": 6,
        "action.gripper_close": 1,
        "action.base_motion": 4,
        "action.control_mode": 1,
    }
    STATE_DIM = 18
    ACTION_DIM = 15
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    @classmethod
    def validate_key_dims(cls):
        assert sum(cls.state_key_dims.values()) == cls.STATE_DIM
        assert sum(cls.action_key_dims.values()) == cls.ACTION_DIM
        assert list(cls.state_key_dims) == cls.state_keys, "state key order drift"
        assert list(cls.action_key_dims) == cls.action_keys, "action key order drift"

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "mean_std" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])


RoboCasaKitchenR6DataConfig.validate_key_dims()


class RoboCasaKitchenR6PointmapDataConfig(RoboCasaKitchenR6DataConfig):
    """The r6 recipe plus a ROBOT-BASE-frame pointmap per camera view.

    * ``make_dataset()`` (the ``lerobot_datasets.make_LeRobotSingleDataset`` opt-in hook)
      swaps in :class:`PointmapPairedDataset`: RGB / state / action from the original
      dataset, depth + camera features from the episode-1:1 pointmap dataset
      (``datasets.vla_data.pointmap_data_root`` in the yaml).
    * ``transform()`` puts :class:`PointmapDerive` FIRST: ``frame='base'`` consumes the
      RAW world base pose (``state.base_position`` / ``state.base_rotation``, xyzw) which
      the parent's mean_std StateActionTransform would otherwise normalize away.

    View order of ``pm_views`` mirrors ``video_keys`` -- it defines the V axis of
    ``sample["pointmap"]``.
    """

    pm_views = {
        view: {
            "depth": f"depth.{view}",
            "intrinsics": f"state.{view}_camera_intrinsics",
            "extrinsics": f"state.{view}_camera_extrinsics",
        }
        for view in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
    }

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.pointmap import PointmapDerive

        parent = super().transform()
        return ComposedModalityTransform(
            transforms=[PointmapDerive(views=self.pm_views, frame="base"), *parent.transforms]
        )

    def make_dataset(self, dataset_path, modality_configs, transforms, embodiment_tag,
                     video_backend, delete_pause_frame, data_cfg, dataset_name):
        from starVLA.dataloader.gr00t_lerobot.paired_pointmap_dataset import PointmapPairedDataset

        pm_root = (data_cfg or {}).get("pointmap_data_root")
        assert pm_root, (
            "pointmap mixture needs datasets.vla_data.pointmap_data_root in the yaml "
            "(the root of the robocasa-MG_300-pointmap dataset)"
        )
        return PointmapPairedDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            pm_dataset_path=pm_root,
            pm_views=self.pm_views,
            # yaml `pointmap_out_hw: [256, 256]` keeps the native depth resolution so the
            # pointmap patch grid matches the RGB one.
            pm_out_hw=tuple((data_cfg or {}).get("pointmap_out_hw", (128, 128))),
        )


_ROBOT_TAG_R6 = "panda_omron_robocasa_kitchen_r6"
_ROBOT_TAG_R6_POINTMAP = "panda_omron_robocasa_kitchen_r6_pointmap"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TAG_R6: RoboCasaKitchenR6DataConfig(),
    _ROBOT_TAG_R6_POINTMAP: RoboCasaKitchenR6PointmapDataConfig(),
}

# Dataset names are relative to the yaml's ``data_root_dir``. The pointmap mixture
# additionally needs ``pointmap_data_root`` (depth + camera features).
DATASET_NAMED_MIXTURES = {
    "robocasa_kitchen_mg300_r6": [("robocasa-MG_300", 1.0, _ROBOT_TAG_R6)],
    "robocasa_kitchen_mg300_r6_pointmap": [("robocasa-MG_300", 1.0, _ROBOT_TAG_R6_POINTMAP)],
}
