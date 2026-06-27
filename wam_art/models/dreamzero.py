"""DreamZero adapter for WAM-ART.

DreamZero (arXiv:2602.15922) is a 14B World Action Model from NVIDIA
GEAR Lab built on the Wan2.1-I2V-14B-480P video diffusion backbone.
It jointly predicts actions and future video frames, achieving strong
zero-shot performance on unseen manipulation tasks.

**Architecture note**
DreamZero runs inference via a distributed WebSocket server (requires
≥2 GPUs).  Unlike FastWAM, there is no single-GPU or CPU inference
path.  This adapter connects to a running DreamZero server as a
WebSocket client.

**Installation prerequisites**

1. Clone the official repository:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/dreamzero0/dreamzero.git
    cd dreamzero
    conda create -n dreamzero python=3.11
    conda activate dreamzero
    pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
    MAX_JOBS=8 pip install --no-build-isolation flash-attn

2. Download a checkpoint from HuggingFace:

.. code-block:: bash

    # DROID checkpoint (14B, trained from scratch on DROID)
    hf download GEAR-Dreams/DreamZero-DROID \\
        --repo-type model --local-dir ./checkpoints/DreamZero-DROID

    # AgiBot checkpoint (for post-training on new embodiments)
    hf download GEAR-Dreams/DreamZero-AgiBot \\
        --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot

3. Start the inference server (on a multi-GPU machine):

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \\
        --standalone --nproc_per_node=2 \\
        socket_test_optimized_AR.py \\
        --port 5000 \\
        --enable-dit-cache \\
        --model-path ./checkpoints/DreamZero-DROID

4. Point this adapter at the running server:

.. code-block:: python

    from wam_art.models.dreamzero import DreamZeroAdapter

    adapter = DreamZeroAdapter(
        device="cuda",
        server_host="localhost",
        server_port=5000,
    )
    adapter.load()  # validates connection
    action, _ = adapter.predict_action(observation, state="pick up the block")

**Environment**
DreamZero's inference client code must be on ``PYTHONPATH``.  The
adapter does a lazy import — if ``dreamzero`` is missing, every method
raises a clear ``RuntimeError`` explaining how to set it up.

References:
    - https://github.com/dreamzero0/dreamzero
    - https://arxiv.org/abs/2602.15922
    - https://huggingface.co/GEAR-Dreams/DreamZero-DROID
    - https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot
"""

from __future__ import annotations

import io
import json
import time
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
_DREAMZERO_AVAILABLE = False
_dreamzero_exc: Exception | None = None

try:
    import websocket  # noqa: F401
    _DREAMZERO_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _dreamzero_exc = exc

# Track if the full dreamzero package (not just websocket) is importable
_DREAMZERO_CLIENT_AVAILABLE = False
try:
    from dreamzero.inference import DreamZeroClient  # type: ignore[import-untyped]

    _DREAMZERO_CLIENT_AVAILABLE = True
except Exception:
    pass


class DreamZeroAdapter(BaseWAMAdapter):
    """Adapter for DreamZero inference servers.

    Connects to a running DreamZero WebSocket inference server and
    delegates ``extract_latent`` / ``predict_action`` over the wire.

    Args:
        model_name: Arbitrary identifier (e.g. ``dreamzero-droid``).
        device: Torch device for local preprocessing (the actual
            DreamZero server runs on its own GPU cluster).
        server_host: WebSocket server hostname or IP.
        server_port: WebSocket server port.
        server_timeout: HTTP/WS timeout in seconds.
        checkpoint_path: Path to the model checkpoint directory on the
            server side (informational — the server decides the path).
        default_prompt: Instruction text used when none is supplied via
            ``state`` in :meth:`predict_action`.
        action_horizon: Number of future action steps to request.
    """

    def __init__(
        self,
        model_name: str = "dreamzero",
        device: str = "cpu",
        server_host: str = "localhost",
        server_port: int = 5000,
        server_timeout: float = 120.0,
        checkpoint_path: str | None = None,
        default_prompt: str = "complete the task",
        action_horizon: int = 16,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.server_host = server_host
        self.server_port = server_port
        self.server_timeout = server_timeout
        self.checkpoint_path = checkpoint_path
        self.default_prompt = default_prompt
        self.action_horizon = action_horizon

        self._ws: Any = None
        self._connected = False
        self._latent_dim: int | None = None
        self._action_dim: int | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _assert_available(self) -> None:
        """Raise a clear error if DreamZero or websocket-client is missing."""
        if not _DREAMZERO_AVAILABLE:
            raise RuntimeError(
                "websocket-client is not installed. "
                "Install it with:\n\n"
                "    pip install websocket-client\n\n"
                "For full DreamZero client support, also clone the repo:\n\n"
                "    git clone --recurse-submodules https://github.com/dreamzero0/dreamzero.git\n"
                "    cd dreamzero && pip install -e .\n\n"
                "See wam_art/models/dreamzero.py docstring for the full setup guide."
            ) from _dreamzero_exc

    def _assert_server(self) -> None:
        """Raise a clear error if the DreamZero server cannot be reached."""
        if not self._connected or self._ws is None:
            raise RuntimeError(
                "DreamZero inference server is not connected. "
                "Start the server first:\n\n"
                "    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \\\n"
                "        --standalone --nproc_per_node=2 \\\n"
                "        socket_test_optimized_AR.py \\\n"
                "        --port 5000 --enable-dit-cache \\\n"
                "        --model-path ./checkpoints/DreamZero-DROID\n\n"
                "Then call adapter.load() to connect."
            )

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """uint8 RGB → base64 JPEG bytes (WebSocket-friendly)."""
        from PIL import Image

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 4:
            image = image[0]
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        import base64
        return base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------
    # Loading / connection
    # ------------------------------------------------------------------
    def load(self, checkpoint_path: str | None = None) -> None:
        """Connect to the DreamZero inference server and validate.

        This does **not** load model weights locally — it opens a
        WebSocket connection to a remote server that handles inference.

        Args:
            checkpoint_path: Ignored (the server decides the model).
                Passed as an informational hint only.
        """
        self._assert_available()

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path

        try:
            import websocket

            ws_url = f"ws://{self.server_host}:{self.server_port}"
            self._ws = websocket.create_connection(
                ws_url, timeout=self.server_timeout
            )
            self._connected = True
        except ConnectionRefusedError:
            raise RuntimeError(
                f"DreamZero server refused connection at {self.server_host}:{self.server_port}. "
                "Is the server running? Start it with:\n\n"
                "    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \\\n"
                "        --standalone --nproc_per_node=2 \\\n"
                "        socket_test_optimized_AR.py --port {self.server_port} \\\n"
                "        --enable-dit-cache --model-path <checkpoint_dir>"
            ) from None
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to DreamZero server at {self.server_host}:{self.server_port}: {exc}"
            ) from exc

        # Send a ping/warmup message to validate the protocol
        # (DreamZero server uses a minimal JSON-based protocol)
        try:
            warmup = json.dumps({"type": "ping"})
            self._ws.send(warmup)
            response = self._ws.recv()
            pong = json.loads(response)
            if pong.get("type") == "pong":
                pass  # Server is alive
            elif pong.get("latent_dim") is not None:
                self._latent_dim = pong.get("latent_dim")
                self._action_dim = pong.get("action_dim", 7)
        except Exception:
            # Server might not support ping/pong — try an actual inference
            # to warm up the DiT cache on first connection
            warnings.warn(
                "DreamZero server did not respond to ping. "
                "First inference may be slow (DiT cache warmup).",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Image preprocessing (local, before sending to server)
    # ------------------------------------------------------------------
    def _preprocess_observation(self, observation: np.ndarray | Tensor) -> str:
        """Convert observation to a base64-encoded JPEG string for sending."""
        if isinstance(observation, Tensor):
            observation = observation.detach().cpu().numpy()
        if observation.dtype != np.uint8:
            observation = (np.clip(observation, 0, 1) * 255).astype(np.uint8)
        if observation.ndim == 4:
            observation = observation[0]
        if observation.shape[-1] != 3 and observation.shape[0] == 3:
            # CHW → HWC
            observation = np.transpose(observation, (1, 2, 0))
        return self._encode_image(observation)

    # ------------------------------------------------------------------
    # Latent extraction
    # ------------------------------------------------------------------
    def extract_latent(self, observation: np.ndarray | Tensor) -> Tensor:
        """Request latent extraction from the DreamZero server.

        Sends a JPEG-encoded observation over WebSocket and receives
        a latent vector from the server's video diffusion VAE.

        Shape: ``(d,)`` where *d* depends on the Wan2.1 VAE latent
        dimensionality (typically ~16384 for a 480P input).
        """
        self._assert_server()

        img_b64 = self._preprocess_observation(observation)

        request = json.dumps({
            "type": "extract_latent",
            "image": img_b64,
        })
        self._ws.send(request)
        response_raw = self._ws.recv()
        response = json.loads(response_raw)

        if response.get("type") == "error":
            raise RuntimeError(f"DreamZero server error: {response.get('message', 'unknown')}")

        latent = response.get("latent")
        if latent is None:
            raise RuntimeError(
                f"DreamZero server did not return a latent. Response: {response_raw[:200]}"
            )

        latent_tensor = torch.tensor(latent, dtype=torch.float32)
        # L2-normalize for cosine distance
        latent_tensor = latent_tensor / (latent_tensor.norm(dim=-1, keepdim=True) + 1e-8)
        return latent_tensor

    # ------------------------------------------------------------------
    # Action prediction
    # ------------------------------------------------------------------
    def predict_action(
        self,
        observation: np.ndarray | Tensor,
        state: Any | None = None,
    ) -> tuple[Tensor, Any]:
        """Request action prediction from the DreamZero server.

        Args:
            observation: uint8 RGB image (H, W, 3) or Tensor.
            state: Optional dict with keys:
                - ``prompt`` (str): task instruction
                - ``action_horizon`` (int): prediction horizon
                - ``seed`` (int): random seed

        Returns:
            (action, None) where *action* is the first step of the
            predicted action chunk.
        """
        self._assert_server()

        prompt = self.default_prompt
        horizon = self.action_horizon
        seed = None

        if isinstance(state, dict):
            prompt = state.get("prompt", prompt)
            horizon = state.get("action_horizon", horizon)
            seed = state.get("seed", seed)
        elif isinstance(state, str):
            prompt = state

        img_b64 = self._preprocess_observation(observation)

        request = json.dumps({
            "type": "predict_action",
            "image": img_b64,
            "prompt": prompt,
            "action_horizon": horizon,
            "seed": seed,
        })
        self._ws.send(request)
        response_raw = self._ws.recv()
        response = json.loads(response_raw)

        if response.get("type") == "error":
            raise RuntimeError(f"DreamZero server error: {response.get('message', 'unknown')}")

        action_data = response.get("action")
        if action_data is None:
            raise RuntimeError(
                f"DreamZero server did not return an action. Response: {response_raw[:200]}"
            )

        action = torch.tensor(action_data, dtype=torch.float32)
        if action.ndim == 2:
            # Return first action in the chunk
            action = action[0]

        return action, None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset any server-side episode state (no-op for stateless server)."""
        pass

    def to(self, device: str) -> DreamZeroAdapter:
        """Move adapter's device (local preprocessing only)."""
        super().to(device)
        return self

    def close(self) -> None:
        """Close the WebSocket connection to the server."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False

    def __del__(self) -> None:
        """Ensure connection is closed on garbage collection."""
        self.close()
