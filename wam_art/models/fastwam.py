"""FastWAM adapter for WAM-ART.

FastWAM (arXiv:2603.16666) is a world action model based on the Wan2.2
video diffusion backbone.  It jointly predicts future video frames and
robot actions, evaluated on LIBERO and RoboTwin benchmarks.

**Installation prerequisites**

1. Clone the official repository alongside WAM-ART and install it:

.. code-block:: bash

    git clone https://github.com/yuantianyuan01/FastWAM.git
    cd FastWAM
    pip install -e .

2. Pre-process the ActionDiT backbone **once** (requires GPU):

.. code-block:: bash

    mkdir -p checkpoints
    export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
    python scripts/preprocess_action_dit_backbone.py \
        --model-config configs/model/fastwam.yaml \
        --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
        --device cuda --dtype bfloat16

3. Download a released checkpoint from HuggingFace, e.g.:

.. code-block:: bash

    huggingface-cli download yuanty/fastwam \
        libero_uncond_2cam224.pt \
        libero_uncond_2cam224_dataset_stats.json \
        --local-dir ./checkpoints/fastwam_release

**Environment**
FastWAM expects its repo root on ``PYTHONPATH``.  The adapter does a
lazy import of ``fastwam`` — if it is missing, every method raises a
clear ``RuntimeError`` explaining how to set it up.

**Latent extraction strategy**
We encode the input image through the Wan2.2 VAE and flatten the
resulting spatial latent.  This is the cheapest policy-specific
representation available without running the full diffusion forward
pass.  A future upgrade could extract the video-expert *pre-DiT* tokens
(even richer semantically) at the cost of an additional transformer pass.

References:
    - https://github.com/yuantianyuan01/FastWAM
    - https://huggingface.co/yuanty/fastwam
    - https://arxiv.org/abs/2603.16666
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wam_art.models.base import BaseWAMAdapter

# ---------------------------------------------------------------------------
# Lazy dependency flag
# ---------------------------------------------------------------------------
_FASTWAM_AVAILABLE = False
_fastwam_exc: Exception | None = None

try:
    from fastwam.models.wan22.fastwam import FastWAM
    from omegaconf import OmegaConf

    _FASTWAM_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _fastwam_exc = exc


class FastWAMAdapter(BaseWAMAdapter):
    """Adapter for FastWAM checkpoints.

    Args:
        model_name: Arbitrary identifier (e.g. ``fastwam-libero``).
        device: Torch device.
        checkpoint_path: Path to a **fine-tuned** ``.pt`` checkpoint
            (e.g. ``checkpoints/fastwam_release/libero_uncond_2cam224.pt``).
        cfg_path: Path to the Hydra model YAML config (default:
            ``FastWAM/configs/model/fastwam.yaml``).
        dataset_stats_path: Optional path to a ``dataset_stats.json``
            produced during FastWAM training.  When provided, the
            adapter applies the same mean/std normalisation the model
            saw during training.
        action_horizon: Number of future action steps the model should
            predict at inference time (default: 16).
        num_inference_steps: Diffusion denoising steps for action
            generation (default: 20; lower = faster, higher = better).
        default_prompt: Instruction text used when none is supplied via
            ``state`` in :meth:`predict_action`.
    """

    def __init__(
        self,
        model_name: str = "fastwam",
        device: str = "cpu",
        checkpoint_path: str | None = None,
        cfg_path: str | None = None,
        dataset_stats_path: str | None = None,
        action_horizon: int = 16,
        num_inference_steps: int = 20,
        default_prompt: str = "pick up the object",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.checkpoint_path = checkpoint_path
        self.cfg_path = cfg_path
        self.dataset_stats_path = dataset_stats_path
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.default_prompt = default_prompt

        self._model: FastWAM | None = None
        self._cfg: Any | None = None
        self._norm_mean: Tensor | None = None
        self._norm_std: Tensor | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _assert_available(self) -> None:
        if not _FASTWAM_AVAILABLE:
            raise RuntimeError(
                "FastWAM is not importable.  Follow the setup steps in "
                "wam_art/models/fastwam.py docstring, then restart Python."
            ) from _fastwam_exc

    @staticmethod
    def _resolve_cfg_path(cfg_path: str | None) -> str:
        """Return an absolute path to a FastWAM Hydra config file."""
        if cfg_path is not None:
            p = Path(cfg_path)
            if p.exists():
                return str(p.resolve())
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        # Heuristic: walk up from the FastWAM package to repo root
        try:
            import fastwam

            repo_root = Path(fastwam.__file__).resolve().parent.parent.parent
        except Exception:
            repo_root = Path(".")
        candidate = repo_root / "configs" / "model" / "fastwam.yaml"
        if candidate.exists():
            return str(candidate.resolve())
        raise FileNotFoundError(
            "cfg_path not provided and auto-detection failed. "
            "Please pass cfg_path=/absolute/path/to/fastwam.yaml"
        )

    def _load_dataset_stats(self) -> None:
        """Optional: load image mean / std from FastWAM training stats."""
        if self.dataset_stats_path is None:
            return
        p = Path(self.dataset_stats_path)
        if not p.exists():
            warnings.warn(f"dataset_stats_path not found: {p}", stacklevel=2)
            return
        try:
            import json

            stats = json.loads(p.read_text())
            image_stats = stats.get("images", stats)  # support both nesting styles
            mean = image_stats.get("mean", [0.5, 0.5, 0.5])
            std = image_stats.get("std", [0.5, 0.5, 0.5])
            self._norm_mean = torch.tensor(mean, dtype=torch.float32)
            self._norm_std = torch.tensor(std, dtype=torch.float32)
        except Exception as exc:
            warnings.warn(f"Failed to load dataset stats: {exc}", stacklevel=2)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, checkpoint_path: str | None = None) -> None:
        """Instantiate FastWAM, load weights, and ready for inference.

        Args:
            checkpoint_path: Overrides the path given at construction.
        """
        self._assert_available()

        ckpt = checkpoint_path or self.checkpoint_path
        if ckpt is None:
            raise ValueError(
                "checkpoint_path must be provided to load FastWAM. "
                "Example: adapter.load('checkpoints/fastwam_release/libero_uncond_2cam224.pt')"
            )

        cfg_path = self._resolve_cfg_path(self.cfg_path)
        self._cfg = OmegaConf.load(cfg_path)
        model_cfg = self._cfg.model if hasattr(self._cfg, "model") else self._cfg

        action_dit_path = model_cfg.get("action_dit_pretrained_path")
        skip_dit_load = bool(model_cfg.get("skip_dit_load_from_pretrain", False))
        if action_dit_path is None and not skip_dit_load:
            raise ValueError(
                "model config is missing 'action_dit_pretrained_path'. "
                "Run scripts/preprocess_action_dit_backbone.py first."
            )

        # FastWAM instantiates the full Wan2.2 backbone + ActionDiT head
        self._model = FastWAM.from_wan22_pretrained(
            device=self.device,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            model_id=model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
            tokenizer_model_id=model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
            tokenizer_max_len=model_cfg.get("tokenizer_max_len", 128),
            load_text_encoder=bool(model_cfg.get("load_text_encoder", False)),
            proprio_dim=model_cfg.get("proprio_dim", None),
            redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
            video_dit_config=OmegaConf.to_container(
                model_cfg.get("video_dit_config", {}), resolve=True
            ),
            action_dit_config=OmegaConf.to_container(
                model_cfg.get("action_dit_config", {}), resolve=True
            ),
            action_dit_pretrained_path=action_dit_path,
            skip_dit_load_from_pretrain=skip_dit_load,
            mot_checkpoint_mixed_attn=bool(model_cfg.get("mot_checkpoint_mixed_attn", True)),
            video_train_shift=model_cfg.get("video_scheduler", {}).get("train_shift", 5.0),
            video_infer_shift=model_cfg.get("video_scheduler", {}).get("infer_shift", 5.0),
            video_num_train_timesteps=model_cfg.get("video_scheduler", {}).get("num_train_timesteps", 1000),
            action_train_shift=model_cfg.get("action_scheduler", {}).get("train_shift", 5.0),
            action_infer_shift=model_cfg.get("action_scheduler", {}).get("infer_shift", 5.0),
            action_num_train_timesteps=model_cfg.get("action_scheduler", {}).get("num_train_timesteps", 1000),
            loss_lambda_video=model_cfg.get("loss", {}).get("lambda_video", 1.0),
            loss_lambda_action=model_cfg.get("loss", {}).get("lambda_action", 1.0),
        )

        # Load fine-tuned weights
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self._model.load_checkpoint(str(ckpt_path))
        self._model.eval()

        self._load_dataset_stats()

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def _preprocess_image(self, observation: np.ndarray | Tensor) -> Tensor:
        """ observ → CHW float32 tensor, resized to multiple of 16. """
        x = self._obs_to_tensor(observation)
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, 3, H, W)

        _, _, h, w = x.shape
        # FastWAM expects H,W multiples of 16
        new_h = (h + 15) // 16 * 16
        new_w = (w + 15) // 16 * 16
        if h != new_h or w != new_w:
            x = torch.nn.functional.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        if self._norm_mean is not None and self._norm_std is not None:
            mean = self._norm_mean.to(x.device).view(1, 3, 1, 1)
            std = self._norm_std.to(x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
        return x

    # ------------------------------------------------------------------
    # Latent extraction
    # ------------------------------------------------------------------
    def extract_latent(self, observation: np.ndarray | Tensor) -> Tensor:
        """Return a VAE latent for the observation.

        Encodes the image through the Wan2.2 VAE, flattens the spatial
        latent, and L2-normalises it.

        Shape: ``(d,)`` where *d* depends on image size and VAE channel
        down-sampling factors (typically ~ for a 224×224 image).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        x = self._preprocess_image(observation).to(
            device=self.device, dtype=self._model.torch_dtype
        )

        with torch.no_grad():
            # FastWAM's private helper; returns a single latent tensor
            z = self._model._encode_input_image_latents_tensor(x[0])
            flat = z.flatten(1).squeeze(0)

        flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        return flat

    # ------------------------------------------------------------------
    # Action prediction
    # ------------------------------------------------------------------
    def predict_action(
        self,
        observation: np.ndarray | Tensor,
        state: Any | None = None,
    ) -> tuple[Tensor, Any]:
        """Predict an action chunk via FastWAM inference.

        Args:
            observation: uint8 RGB image (H, W, 3).
            state: Optional dict with keys:
                - ``prompt`` (str): task instruction
                - ``proprio`` (Tensor): proprioception vector [D] or [1, D]
                - ``num_inference_steps`` (int): diffusion steps
                - ``action_horizon`` (int): prediction horizon
                - ``seed`` (int): random seed for deterministic sampling

        Returns:
            (action, None) where *action* is the first step of the
            predicted action chunk (the immediate action to execute).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        x = self._preprocess_image(observation).to(
            device=self.device, dtype=self._model.torch_dtype
        )

        prompt = self.default_prompt
        num_steps = self.num_inference_steps
        horizon = self.action_horizon
        seed = None
        proprio: Tensor | None = None

        if isinstance(state, dict):
            prompt = state.get("prompt", prompt)
            num_steps = state.get("num_inference_steps", num_steps)
            horizon = state.get("action_horizon", horizon)
            seed = state.get("seed", seed)
            proprio_raw = state.get("proprio", None)
            if proprio_raw is not None:
                proprio = self._obs_to_tensor(proprio_raw)
                if proprio.ndim == 1:
                    proprio = proprio.unsqueeze(0)

        with torch.no_grad():
            result = self._model.infer_action(
                prompt=prompt,
                input_image=x[0],
                action_horizon=horizon,
                proprio=proprio,
                num_inference_steps=num_steps,
                seed=seed,
            )

        action = result["action"]  # (action_horizon, action_dim)
        # Return the *first* action as the immediate control signal
        first_action = action[0].detach().cpu().float()
        return first_action, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _obs_to_tensor(observation: np.ndarray | Tensor) -> Tensor:
        if isinstance(observation, np.ndarray):
            if observation.dtype != np.uint8:
                observation = np.clip(observation, 0, 255).astype(np.uint8)
            if observation.ndim == 3:
                arr = observation.astype(np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                return torch.from_numpy(arr)
            elif observation.ndim == 4:
                arr = observation.astype(np.float32) / 255.0
                arr = np.transpose(arr, (0, 3, 1, 2))
                return torch.from_numpy(arr)
        if isinstance(observation, Tensor):
            if observation.ndim == 3:
                observation = observation.unsqueeze(0)
            if observation.shape[1] != 3 and observation.shape[-1] == 3:
                observation = observation.permute(0, 3, 1, 2)
            return observation.float().clamp(0, 1)
        raise TypeError(f"Unexpected observation type: {type(observation)}")

    def reset(self) -> None:
        pass

    def to(self, device: str) -> FastWAMAdapter:
        super().to(device)
        if self._model is not None:
            self._model.to(device)
        return self
