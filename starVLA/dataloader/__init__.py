import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )

        num_workers = int(vla_dataset_cfg.get("num_workers", 4))

        # collate_vlm_inputs: run the Qwen-VL preprocessing (chat template, tokenization,
        # image resize/normalize) inside the dataloader workers instead of on the main
        # process, where it would sit serially in front of the GPU forward. The collator
        # also packs the per-sample pointmaps when collate_pointmap is set.
        # See qwenvl_collator.py.
        if bool(vla_dataset_cfg.get("collate_vlm_inputs", False)):
            from starVLA.dataloader.qwenvl_collator import QwenVLBatchCollator

            collate_pointmap = bool(vla_dataset_cfg.get("collate_pointmap", False))
            active_collate_fn = QwenVLBatchCollator(
                base_vlm=cfg.framework.qwenvl.base_vlm,
                cot_prompt=vla_dataset_cfg.get("CoT_prompt", None),
                instruction_normalize=bool(vla_dataset_cfg.get("instruction_normalize", False)),
                pack_pointmap=collate_pointmap,
                enable_thinking=cfg.framework.qwenvl.get("enable_thinking", None),
                pointmap_origin_tag=str(vla_dataset_cfg.get("pointmap_origin_tag", "") or ""),
            )
            print(
                f"[dataloader] collate_vlm_inputs=true -- Qwen-VL preprocessing moved into "
                f"{num_workers} dataloader workers ({cfg.framework.qwenvl.base_vlm}); "
                f"collate_pointmap={collate_pointmap}",
                flush=True,
            )
        else:
            active_collate_fn = collate_fn

        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": active_collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
