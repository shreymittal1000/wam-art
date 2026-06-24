"""FastWAM adapter for WAM-ART.

FastWAM (arXiv:2603.16666) is a world action model based on the Wan2.2
video diffusion backbone.  It jointly predicts future video frames and
robot actions, evaluated on LIBERO and RoboTwin benchmarks.

**Installation prerequisites:**

.. code-block:: bash

    # 1. Clone the official repository alongside WAM-ART
    git clone https://github.com/yuantianyuan01/FastWAM.git
    cd FastWAM
    pip install -e .

    # 2. Download checkpoints from HuggingFace
    huggingface-cli download yuanty/fastwam \
        libero_uncond_2cam224.pt \
        libero_uncond_2cam224_dataset_stats.json \
        --local-dir ./checkpoints/fastwam_release

    # 3. (Optional) Pre-process the ActionDiT backbone
    python scripts/preprocess_action_dit_backbone.py \
        --model-config configs/model/fastwam.yaml \
        --output checkpoints/ActionDiT.pt --device cuda --dtype bfloat16

**Environment:**
FastWAM expects its repo root in ``PYTHONPATH``.  When this adapter is
instantiated, it attempts an *optional* import from ``fastwam`` — if the
module is missing, every method raises ``RuntimeError`` with a setup
reminder.

**Latent extraction strategy:**
Currently returns **VAE-encoded latent** of the input image.  A future
upgrade can extract the DiT hidden state after a conditioning forward
pass ( richer semantically but more expensive).

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
    import fastwam.runtime as fw_runtime
    from omegaconf import OmegaConf

    _FASTWAM_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _fastwam_exc = exc


class FastWAMAdapter(BaseWAMAdapter):
    """Adapter for FastWAM checkpoints.

    Args:
        model_name: Arbitrary identifier (e.g. ``fastwam-libero``).
        device: Torch device.
        checkpoint_path: Path to a ``.pt`` checkpoint (e.g.
            ``checkpoints/fastwam_release/libero_uncond_2cam224.pt``).
        task_config_path: Path to the Hydra task YAML config that
            matches the checkpoint (e.g.
            ``configs/task/libero_uncond_2cam224_1e-4.yaml``).
        default_prompt: Instruction text used when no prompt is supplied.
    """

    def __init__(
        self,
        model_name: str = "fastwam",
        device: str = "cpu",
        checkpoint_path: str | None = None,
        task_config_path: str | None = None,
        default_prompt: str = "pick up the object",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.checkpoint_path = checkpoint_path
        self.task_config_path = task_config_path
        self.default_prompt = default_prompt

        self._model: Any | None = None
        self._cfg: Any | None = None
        self._vae: Any | None = None

    # -------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------
    def load(self, checkpoint_path: str | None = None) -> None:
        """Instantiate FastWAM and load weights.

        Args:
            checkpoint_path: Overrides the path given at construction.
        """
        if not _FASTWAM_AVAILABLE:
            raise RuntimeError(
                "FastWAM is not importable.  Follow the setup steps in "
                "wam_art/models/fastwam.py docstring, then restart Python."
            ) from _fastwam_exc

        ckpt = checkpoint_path or self.checkpoint_path
        if ckpt is None:
            raise ValueError("checkpoint_path must be provided to load FastWAM.")

        # Resolve task config
        cfg_path = self.task_config_path
        if cfg_path is None:
            # Attempt sibling-directory heuristic inside the FastWAM repo
            repo_root = Path(fw_runtime.__file__).resolve().parent.parent.parent
            default_cfg = repo_root / "configs" / "model" / "fastwam.yaml"
            if default_cfg.exists():
                cfg_path = str(default_cfg)
            else:
                raise ValueError(
                    "task_config_path not provided and auto-detection failed."
                )

        # Load Hydra config and instantiate model
        self._cfg = OmegaConf.load(cfg_path)
        # FastWAM runtime factory requires model + scheduler configs
        self._model = fw_runtime.create_fastwam(
            model_id=self._cfg.model.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
            tokenizer_model_id=self._cfg.model.get(
                "tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"
            ),
            video_dit_config=self._cfg.model.get("video_dit_config", {}),
            action_dit_config=self._cfg.model.get("action_dit_config", {}),
            action_dit_pretrained_path=self._cfg.model.get(
                "action_dit_pretrained_path", None
            ),
            video_scheduler=self._cfg.model.get("video_scheduler", {}),
            action_scheduler=self._cfg.model.get("action_scheduler", {}),
            loss=self._cfg.model.get("loss", {}),
            model_dtype=torch.bfloat16,
            device=self.device,
        )

        # Load fine-tuned weights
        ckpt_path = Path(ckpt)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self._model.load_state_dict(state)
            self._model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self._vae = self._model.vae

    # -------------------------------------------------------------------
    # Latent extraction
    # -------------------------------------------------------------------
    def extract_latent(self, observation: np.ndarray | Tensor) -> Tensor:
        """Return a VAE latent for the observation.

        FastWAM is a video diffusion model; for single-image observations
        we encode through the Wan2.2 VAE and return the spatial latent
        vector (flattened and L2-normalised).

        Shape: ``(d,)`` where *d* depends on image size and VAE channel
        down-sampling factors.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        x = self._obs_to_tensor(observation).to(
            device=self.device, dtype=self._model.torch_dtype
        )
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            # FastWAM VAE expects a list of 3D tensors [C, T, H, W]
            # For a single image, T=1.
            image = x[:, :, None, :, :]  # (1, 3, 1, H, W)
            latent = self._vae.encode([image[0]], device=self.device)
            if isinstance(latent, list):
                latent = latent[0]
            latent = latent.unsqueeze(0)
            # Flatten spatial dims + channels -> single vector
            flat = latent.flatten(1)  # (1, d)

        flat = flat.squeeze(0)
        flat = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        return flat

    # -------------------------------------------------------------------
    # Action prediction
    # -------------------------------------------------------------------
    def predict_action(
        self,
        observation: np.ndarray | Tensor,
        state: Any | None = None,
    ) -> tuple[Tensor, Any]:
        """Predict an action chunk via FastWAM inference.

        Args:
            observation: uint8 RGB image.
            state: Optional dict with keys ``prompt`` (str) and
                ``num_frames`` / ``num_inference_steps``.

        Returns:
            (action, None) where *action* is the first step of the
            predicted action chunk (the immediate action to execute).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        x = self._obs_to_tensor(observation).to(
            device=self.device, dtype=self._model.torch_dtype
        )
        if x.ndim == 3:
            x = x.unsqueeze(0)

        prompt = self.default_prompt
        num_frames = 17
        num_steps = 50
        if isinstance(state, dict):
            prompt = state.get("prompt", prompt)
            num_frames = state.get("num_frames", num_frames)
            num_steps = state.get("num_inference_steps", num_steps)

        # Encode text prompt
        prompt_emb, mask = self._model.encode_prompt(prompt)

# Encode input image latent (currently unused; scaffold for future)
        self._model._encode_input_image_latents_tensor(x[0])

        # TODO: run the full diffusion denoising loop to extract the
        # action expert output.  FastWAM inference is non-trivial and
        # should mirror the logic in ``experiments/libero/run_libero_*.py``.
        # For now we return a zero tensor placeholder so that the pipeline
        # can be wired end-to-end.
        warnings.warn(
            "FastWAMAdapter.predict_action is a scaffold. "
            "Full diffusion-based action extraction is not yet implemented.",
            stacklevel=2,
        )
        dummy_action = torch.zeros(self._guess_action_dim(), device=self.device)
        return dummy_action, None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _obs_to_tensor(observation: np.ndarray | Tensor) -> Tensor:
        if isinstance(observation, np.ndarray):
            # uint8 HWC → float CHW [0,1]
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
            return observation.float()
        raise TypeError(f"Unexpected observation type: {type(observation)}")

    def _guess_action_dim(self) -> int:
        """Heuristic until proper action extraction is wired."""
        # LIBERO / RoboTwin action spaces are typically 7-8 DoF
        return 7

    def reset(self) -> None:
        pass

    def to(self, device: str) -> FastWAMAdapter:
        super().to(device)
        if self._model is not None:
            self._model.to(device)
        return self
