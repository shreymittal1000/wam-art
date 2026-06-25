"""OpenVLA adapter for WAM-ART.

OpenVLA is a 7B-parameter Vision-Language-Action model (VLA) built on
Llama 2 + DINOv2/SigLIP vision backbone.  Although technically a VLA
rather than a pure World Action Model, its observation-to-action
mapping is a valid target for WAM-ART robustness analysis.

**Installation:**

.. code-block:: bash

    pip install timm==0.9.16 transformers==4.40.1 tokenizers==0.19.1
    # Clone OpenVLA repo for custom model code, or install from source:
    pip install -e git+https://github.com/openvla/openvla.git#egg=openvla

**HuggingFace checkpoint:** ``openvla/openvla-7b``

References:
    - https://github.com/openvla/openvla
    - https://huggingface.co/openvla/openvla-7b
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from wam_art.models.base import BaseWAMAdapter

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "transformers is required for OpenVLAAdapter. "
        "Install with: pip install transformers==4.40.1 timm==0.9.16"
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required for OpenVLAAdapter.") from exc


class OpenVLAAdapter(BaseWAMAdapter):
    """Adapter for OpenVLA models loaded via HuggingFace transformers.

    Args:
        model_name: HF Hub repo id (e.g. ``openvla/openvla-7b``).
        device: Torch device string.
        unnorm_key: Dataset name for action un-normalization stats
            (e.g. ``bridge_orig``).  Required for fine-tuned checkpoints;
            leave *None* for the base model if it was trained on a single
            dataset.
        default_instruction: Fallback instruction used when none is
            provided to ``predict_action``.
    """

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        device: str = "cpu",
        unnorm_key: str | None = "bridge_orig",
        default_instruction: str = "complete the task",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.unnorm_key = unnorm_key
        self.default_instruction = default_instruction
        self._vla: Any | None = None
        self._processor: Any | None = None
        self._image_transform: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded_checkpoint: str | None = None

    # -------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------
    def load(self, checkpoint_path: str | None = None) -> None:
        """Load model weights from HuggingFace Hub or local path.

        Args:
            checkpoint_path: HF Hub repo id or local directory.  If
                *None* the ``model_name`` passed at construction is used.
        """
        path = checkpoint_path or self.model_name
        self._loaded_checkpoint = path

        # OpenVLA requires custom model code; trust_remote_code handles
        # downloading the modeling files from the HF Hub automatically.
        self._processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=True
        )
        self._vla = AutoModelForVision2Seq.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self._vla.eval()
        self._vla.to(self.device)

        # Cache references to internal helpers used for latent extraction.
        # Note: HF PrismaticVisionBackbone does not expose image_transform.
        # We use the processor for preprocessing instead.
        self._image_transform = None

        if hasattr(self._vla, "llm_backbone"):
            self._tokenizer = self._vla.llm_backbone.tokenizer
        elif hasattr(self._vla, "model") and hasattr(self._vla.model, "get_input_embeddings"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor.tokenizer

    # -------------------------------------------------------------------
    # Latent extraction
    # -------------------------------------------------------------------
    def extract_latent(self, observation: np.ndarray | Tensor) -> Tensor:
        """Extract observation encoding from the vision backbone.

        For OpenVLA the latent is defined as **mean-pooled DINOv2
        patch features** (after the vision tower, before the multimodal
        projector).  This is the richest visual representation available
        without entering the text-conditioned LLM path.

        Strategy:
            1. ``observation`` (uint8 HWC) → PIL Image →
               ``pixel_values`` via the model's ``image_transform``.
            2. Run through ``vision_backbone.featurizer``.
            3. Mean pool over spatial patches → L2-normalise.
        """
        if self._vla is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        img = self._to_pil(observation)
        pixel_values = self._preprocess_image(img)

        with torch.no_grad():
            # Use the vision backbone directly if accessible
            vision_mod = self._resolve_vision_module()
            if vision_mod is not None:
                features = vision_mod(pixel_values)  # (1, N_patches, D)
                latent = features.mean(dim=1)          # (1, D)
            else:
                # Fallback: processor-level latent (less expressive)
                inputs = self._processor(images=img, return_tensors="pt")
                latent = inputs["pixel_values"].to(self.device).flatten(1)

        latent = latent.squeeze(0)
        latent = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
        return latent

    # -------------------------------------------------------------------
    # Action prediction
    # -------------------------------------------------------------------
    def predict_action(
        self,
        observation: np.ndarray | Tensor,
        state: Any | None = None,
    ) -> tuple[Tensor, Any]:
        """Predict a single-step action using OpenVLA.

        Args:
            observation: uint8 RGB image (H, W, 3).
            state: Optional instruction string or dict with key
                ``instruction``.  Falls back to ``default_instruction``.

        Returns:
            (action_tensor, None) — action shape depends on the
            dataset the checkpoint was fine-tuned on (typically 6-7 DoF).
        """
        if self._vla is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        img = self._to_pil(observation)

        # OpenVLA natively implements `predict_action` on the model class.
        # Distinguish the two possible API shapes.
        # NOTE: the model's predict_action expects input_ids + pixel_values;
        # we build them via the processor here.
        prompt = self._resolve_instruction(state)
        inputs = self._processor(prompt, images=img, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self._vla.dtype)

        with torch.no_grad():
            if hasattr(self._vla, "predict_action"):
                action = self._vla.predict_action(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    unnorm_key=self.unnorm_key,
                )
            else:
                raise RuntimeError(
                    "Loaded checkpoint does not expose `predict_action`. "
                    "Ensure you are loading an OpenVLA checkpoint, not a base VLM."
                )

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        else:
            action = action.float().to(self.device)

        return action, None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _to_pil(observation: np.ndarray | Tensor) -> Image.Image:
        """Convert a uint8/HWC observation to PIL RGB."""
        if isinstance(observation, Tensor):
            observation = observation.detach().cpu().numpy()
        if observation.dtype != np.uint8:
            observation = (np.clip(observation, 0, 1) * 255).astype(np.uint8)
        if observation.ndim == 4:
            observation = observation[0]
        return Image.fromarray(observation).convert("RGB")

    def _preprocess_image(self, img: Image.Image) -> Tensor:
        """Run the model-specific image transform via the processor."""
        if self._image_transform is not None:
            pixel_values = self._image_transform(img)
            if isinstance(pixel_values, dict):
                pixel_values = pixel_values["pixel_values"]
            if pixel_values.ndim == 3:
                pixel_values = pixel_values[None, ...]
            return pixel_values.to(self.device, dtype=torch.bfloat16)

        # Fallback via HF processor (works for OpenVLA-7b loaded via
        # AutoModelForVision2Seq + trust_remote_code).
        # PrismaticProcessor requires a text argument even for image-only
        # preprocessing, so we pass a dummy string.
        inputs = self._processor("dummy", images=img, return_tensors="pt")
        pv = inputs["pixel_values"]
        if pv.ndim == 3:
            pv = pv[None, ...]
        return pv.to(self.device, dtype=torch.bfloat16)

    def _resolve_vision_module(self) -> Any | None:
        """Return the raw vision backbone forward callable, or None."""
        if hasattr(self._vla, "vision_backbone"):
            return self._vla.vision_backbone
        if hasattr(self._vla, "model") and hasattr(self._vla.model, "vision_backbone"):
            return self._vla.model.vision_backbone
        return None

    def _resolve_instruction(self, state: Any | None) -> str:
        """Extract instruction string from ``state`` or use default."""
        if state is None:
            return self.default_instruction
        if isinstance(state, str):
            return state
        if isinstance(state, dict):
            return state.get("instruction", self.default_instruction)
        return self.default_instruction

    def reset(self) -> None:
        """No episodic state to reset for the underlying VLA."""
        pass

    def to(self, device: str) -> OpenVLAAdapter:
        super().to(device)
        if self._vla is not None:
            self._vla.to(device)
        return self
