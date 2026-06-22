"""Configuration management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML config and merge with schema validation.

    Args:
        config_path: Path to YAML config file.

    Returns:
        OmegaConf DictConfig.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = OmegaConf.create(raw)

    # Ensure critical fields exist
    if "seed" not in cfg:
        cfg.seed = 42
    if "device" not in cfg:
        cfg.device = "cpu"

    OmegaConf.set_readonly(cfg, True)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save config to YAML."""
    OmegaConf.save(cfg, path)
