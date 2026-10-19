#!/usr/bin/env python
"""Download the processed fine-tuning datasets for PointmapVLA from Hugging Face.

Each benchmark needs two LeRobot v3.0 datasets: one with RGB videos, robot
state and actions, and one with the matching depth videos and per-frame camera
parameters from which the loader builds robot-base-frame pointmaps on the fly.
Files are placed exactly where the shipped training configs expect them.

Usage:
    python scripts/download_data.py --bench robocasa
    python scripts/download_data.py --bench libero
    python scripts/download_data.py --bench robocasa libero --dest playground/Datasets
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

# (HF dataset repo, local subdirectory under --dest)
DATASETS = {
    "robocasa": [
        ("DAVIAN-Robotics/robocasa-MG_300-r6", "robocasa_kitchen_mg300_r6/robocasa-MG_300"),
        ("DAVIAN-Robotics/robocasa-MG_300-pointmap", "robocasa-MG_300-pointmap"),
    ],
    "libero": [
        ("DAVIAN-Robotics/libero-r6", "libero_r6"),
        ("DAVIAN-Robotics/libero-r6-pointmap", "libero_r6_pointmap"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", nargs="+", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dest", default="playground/Datasets", help="dataset root used by the configs")
    parser.add_argument("--revision", default=None, help="dataset repo revision (default: latest)")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (lower it if you hit HTTP 429)")
    parser.add_argument("--retries", type=int, default=10, help="attempts per dataset before giving up")
    parser.add_argument("--retry-wait", type=int, default=60, help="seconds to wait between attempts")
    args = parser.parse_args()

    dest = Path(args.dest)
    for bench in args.bench:
        for repo_id, subdir in DATASETS[bench]:
            local_dir = dest / subdir
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{bench}] {repo_id} -> {local_dir}")
            # The LIBERO repos hold thousands of small files and the Hub rate-limits bursts
            # of requests (HTTP 429). snapshot_download resumes, so retry with a pause.
            for attempt in range(1, args.retries + 1):
                try:
                    snapshot_download(
                        repo_id,
                        repo_type="dataset",
                        revision=args.revision,
                        local_dir=str(local_dir),
                        max_workers=args.workers,
                    )
                    break
                except Exception as e:  # network / rate-limit errors
                    if attempt == args.retries:
                        raise
                    print(f"  download interrupted ({type(e).__name__}: {e}); retrying in {args.retry_wait}s "
                          f"({attempt}/{args.retries})")
                    time.sleep(args.retry_wait)
    print("done")


if __name__ == "__main__":
    main()
