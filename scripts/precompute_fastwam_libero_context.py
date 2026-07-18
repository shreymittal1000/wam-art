#!/usr/bin/env python
"""Precompute one FastWAM LIBERO task prompt embedding on CPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from libero.libero import benchmark
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.models.wan22.helpers.loader import (
        _load_registered_model,
        _resolve_configs,
    )
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    prompt = DEFAULT_PROMPT.format(task=task.language)

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=False,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=128,
        clean="whitespace",
    )
    ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
    mask = mask.to(dtype=torch.bool)
    with torch.no_grad():
        context = encoder(ids, mask)
    seq_lens = mask.gt(0).sum(dim=1).long()
    for i, length in enumerate(seq_lens):
        context[i, length:] = 0
    mask = torch.ones_like(mask)

    output = Path(os.path.expanduser(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"context": context.cpu(), "context_mask": mask.cpu(), "prompt": prompt},
        output,
    )
    print(f"Saved {task.language!r} context {tuple(context.shape)} to {output}")


if __name__ == "__main__":
    main()
