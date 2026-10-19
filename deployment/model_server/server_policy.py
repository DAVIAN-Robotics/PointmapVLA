# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

# Serving runs single-process at a small batch size, where the flash-attention
# varlen kernels can return NaNs. The weights are identical under SDPA, so the
# server always prefers SDPA; set STARVLA_FORCE_SDPA=0 explicitly to override.
os.environ.setdefault("STARVLA_FORCE_SDPA", "1")

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch once at startup so that re-running the same
    sequence of requests reproduces the same actions (the flow-matching head
    draws its initial noise from the global torch RNG unless the client passes
    a per-request ``noise_seed``)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info("global RNG seeded with %d", seed)


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    import torch

    # The released checkpoints were trained with TF32 matmuls; serve them the same way.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed = args.seed if args.seed is not None else os.getenv("STARVLA_EVAL_SEED")
    if seed is not None:
        seed_everything(int(seed))

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt,
        device="cuda",
        use_bf16=args.use_bf16,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # =========================================================================
    # !!! TRAIN / TEST CONSISTENCY — READ BEFORE SERVING !!!
    # -------------------------------------------------------------------------
    # This server replays the *training-time* observation contract. The eval
    # client MUST feed observations exactly as the model saw them at TRAIN time,
    # otherwise the success rate silently drops (no error is raised):
    #   - state   : is proprioceptive state used? (use_state) and its dim/order
    #   - img size: resize / crop resolution (e.g. 224x224)
    #   - img num : how many camera views are fed
    #   - img order: the ordering of those camera views
    #   - action normalization: unnorm_key / dataset stats must match training
    # `wrapper.metadata` (logged below and sent at handshake) exposes
    # action_chunk_size / state_keys / action_keys — cross-check these against
    # the training config used to produce `args.ckpt`.
    # =========================================================================
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s — verify eval observations "
        "(state / image size / image count / image order / action normalization) match "
        "the training config. metadata=%s",
        args.ckpt,
        wrapper.metadata,
    )

    # start websocket server; wrapper.metadata is sent at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    logging.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", "--ckpt_path", dest="ckpt", type=str, required=True,
        help="Checkpoint: a run directory (config.yaml + dataset_statistics.json + checkpoints/), "
             "a checkpoint file inside such a directory, or a Hugging Face repo id.",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the global RNGs once at startup (same as STARVLA_EVAL_SEED).")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
