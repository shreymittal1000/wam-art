"""VLM-guided perturbation editors.

Uses vision-capable LLMs (Gemini, GPT-4o, etc.) to plan parametric
corruptions from natural-language instructions.  Currently built on top
of the OpenRouter endpoint so it reuses the existing API key setup.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import warnings
from typing import Any

import numpy as np
import requests
from PIL import Image

from wam_art.editing.corruptions import apply_corruption, list_corruptions
from wam_art.editing.critic import BaseCritic, CriticResult
from wam_art.editing import BaseEditor


class VLMPerturbationEditor(BaseEditor):
    """VLM-driven parametric corruption planner.

    Sends the image + instruction to a vision-capable model and asks it
    to pick the most appropriate parametric corruption from the registry
    together with concrete parameters.  The editor then executes the
    chosen corruption.

    Args:
        factor_name: Human-readable label for logging.
        api_key: OpenRouter (or generic OpenAI-compatible) API key.
            Defaults to ``OPENROUTER_API_KEY`` env var.
        model: Model identifier on the endpoint.
            Default is Gemini 2.5 Flash via OpenRouter.
        base_url: Chat-completions endpoint.
        timeout: HTTP timeout per call.
        max_retries: Retries on transient failures.
        critic: Optional post-edit critic (same interface as
            :class:`RichPerturbationEditor`).
        temperature: Sampling temperature for the VLM planner.
    """

    _CORRUPTION_REGISTRY_HELP: str = (
        "Available corruptions (with relevant parameter ranges):\n"
        "- motion_blur: kernel_size (int, 3-21), angle (float, 0-360)\n"
        "- gaussian_blur: kernel_size (int, 3-21), sigma (float, 0.5-5.0)\n"
        "- occlusion: ratio (float, 0.0-0.5), position ('center', 'random', 'top-left', etc.)\n"
        "- brightness_shift: factor (float, 0.3-1.7)\n"
        "- contrast_shift: factor (float, 0.3-1.7)\n"
        "- saturation_shift: factor (float, 0.0-2.0)\n"
        "- jpeg_compression: quality (int, 10-95)\n"
        "- gaussian_noise: sigma (float, 0.01-0.3)\n"
        "- perspective_warp: magnitude (float, 0.01-0.3)\n"
        "- salt_and_pepper: amount (float, 0.01-0.1)\n"
    )

    _SYSTEM_PROMPT: str = (
        "You are a robotics vision perturbation planner. "
        "Given a camera image and a natural-language instruction, "
        "choose the SINGLE most appropriate parametric corruption from the registry "
        "and return **only** a JSON object with no markdown formatting, "
        "no code fences, and no extra commentary.\n\n"
        "{registry_help}\n\n"
        "Return format (EXACTLY this JSON, nothing else):\n"
        '{{"corruption": "...", "params": {{...}}, "explanation": "..."}}'
    )

    def __init__(
        self,
        factor_name: str,
        api_key: str | None = None,
        model: str = "google/gemini-2.5-flash",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: float = 60.0,
        max_retries: int = 2,
        critic: BaseCritic | None = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(factor_name)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VLMPerturbationEditor requires an API key. "
                "Set OPENROUTER_API_KEY env var or pass api_key=..."
            )
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.critic = critic
        self.temperature = temperature
        self._available_corruptions = set(list_corruptions())

    # ------------------------------------------------------------------
    # Encoding & API helpers (shared pattern with APICritic)
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """uint8 RGB -> base64 PNG data URI."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = __import__("base64").b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def _call_api(self, image: np.ndarray, instruction: str) -> dict[str, Any]:
        """Send image + instruction to the VLM and return raw JSON."""
        system_msg = self._SYSTEM_PROMPT.format(
            registry_help=self._CORRUPTION_REGISTRY_HELP
        )
        user_msg = (
            f"Instruction:\n{instruction}\n\n"
            "Choose the best corruption and return the JSON."
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_image(image)},
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "WAM-ART",
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    warnings.warn(
                        f"VLMPerturbationEditor API call failed (attempt {attempt + 1}): {exc}. Retrying...",
                        stacklevel=3,
                    )

        raise RuntimeError(
            f"VLMPerturbationEditor failed after {self.max_retries + 1} attempts. Last error: {last_exc}"
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Robustly pull a JSON object from model text.

        Handles markdown fences, extra prose, etc.
        """
        # Try direct json first
        text = text.strip()
        if text.startswith("```"):
            # Strip markdown fences
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Try regex extraction of the outermost {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from model response: {exc}\nRaw text:\n{text[:500]}") from exc

    @staticmethod
    def _parse_planner_response(text: str) -> tuple[str, dict[str, Any], str]:
        """Parse planner JSON into (corruption_name, params, explanation)."""
        data = VLMPerturbationEditor._extract_json(text)
        corruption = data.get("corruption")
        params = data.get("params", {})
        explanation = data.get("explanation", "")
        if not isinstance(corruption, str):
            raise ValueError(f"Expected 'corruption' string, got {type(corruption)}")
        return corruption, params, explanation
        explanation = data.get("explanation", "")
        if not isinstance(corruption, str):
            raise ValueError(f"Expected 'corruption' string, got {type(corruption)}")
        return corruption, params, explanation

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------
    def edit(self, image: np.ndarray, instruction: str, **kwargs: Any) -> np.ndarray:
        """Plan and apply a parametric corruption via VLM.

        On API failure or parse failure, returns the original image
        unmodified (fail-open).
        """
        try:
            api_resp = self._call_api(image, instruction)
            message = api_resp["choices"][0]["message"]["content"]
            corruption, params, explanation = self._parse_planner_response(message)
        except Exception as exc:
            raise RuntimeError(
                f"VLMPerturbationEditor planning failed: {exc}. "
                f"Image was not modified."
            ) from exc

        if corruption not in self._available_corruptions:
            warnings.warn(
                f"VLM chose unknown corruption {corruption!r}. "
                f"Available: {self._available_corruptions}. Returning original image.",
                stacklevel=2,
            )
            return image

        edited = apply_corruption(corruption, image, **params)

        if self.critic is not None:
            description = f"{self.factor_name}: {corruption} with {params} — {explanation}"
            result = self.critic.judge(
                edited,
                edit_description=description,
                original_image=image,
            )
            if not result.passes:
                return image

        return edited


class GeminiPerturbationEditor(VLMPerturbationEditor):
    """Convenience alias for VLMPerturbationEditor.

    Uses ``openai/gpt-4o`` by default for stronger planning.
    """

    def __init__(
        self,
        factor_name: str,
        api_key: str | None = None,
        model: str = "openai/gpt-4o",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            factor_name=factor_name,
            api_key=api_key,
            model=model,
            **kwargs,
        )
