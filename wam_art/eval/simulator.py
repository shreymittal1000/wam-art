"""Simulator harness for measuring actual task success in robotics benchmarks.

Provides a pluggable interface so WAM-ART can evaluate on real (or
simulated) robot environments.  The reference implementation wraps LIBERO
(``libero``), a MuJoCo-based manipulation benchmark widely used by
FastWAM and OpenVLA papers.

**Rendering requirements**

LIBERO / robosuite need an off-screen OpenGL context to produce camera
images.  One of the following must be available on the host:

- **EGL** (GPU)   – fastest, requires ``/dev/dri/renderD*`` access.
- **OSMesa** (CPU) – ``apt install libosmesa6-dev`` then set
  ``MUJOCO_GL=osmesa``.
- **Xvfb + GLFW** – ``apt install xvfb``, then run scripts under
  ``xvfb-run -a``.

When no renderer is available, :class:`LiberoSimulator` automatically
falls back to :class:`MockSimulator` with a loud warning so the
pipeline can still be tested end-to-end.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from wam_art.models.base import BaseWAMAdapter


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single simulated episode."""

    success: bool
    steps: int
    total_reward: float
    image: np.ndarray | None  # final observation frame (H, W, 3) uint8
    info: dict[str, Any]


@dataclass(frozen=True)
class TaskResult:
    """Aggregated results for a task under a specific corruption factor."""

    task_name: str
    factor_name: str
    n_episodes: int
    success_rate: float  # measured_failure = 1 - success_rate
    mean_steps: float
    mean_reward: float


# ---------------------------------------------------------------------------
# Base simulator
# ---------------------------------------------------------------------------
class BaseSimulator(ABC):
    """Abstract interface for a robotics benchmark simulator.

    A simulator is responsible for:
    1. Providing nominal camera observations from a defined task.
    2. Executing actions supplied by a WAM adapter.
    3. Reporting episode-level success / failure.
    """

    @abstractmethod
    def reset_task(self, task_id: int | str, seed: int = 0) -> np.ndarray:
        """Reset to a specific task and return the initial observation.

        Args:
            task_id: Task identifier (index or name).
            seed: RNG seed for reproducible initial state.

        Returns:
            Initial RGB image (H, W, 3) uint8.
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute one action in the simulator.

        Args:
            action: Action vector (shape depends on robot / task).

        Returns:
            (next_observation_image, reward, done, info)
        """
        ...

    @abstractmethod
    def run_episode(
        self,
        adapter: BaseWAMAdapter,
        task_id: int | str,
        max_steps: int = 100,
        seed: int = 0,
    ) -> EpisodeResult:
        """Run a complete episode using ``adapter`` to select actions.

        Args:
            adapter: WAM adapter that implements ``predict_action``.
            task_id: Task identifier.
            max_steps: Hard step limit per episode.
            seed: RNG seed.

        Returns:
            EpisodeResult with success flag and final frame.
        """
        ...

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return human-readable task names available in this suite."""
        ...

    def close(self) -> None:
        """Release simulator resources.  Override in subclasses that hold
        heavy MuJoCo / rendering contexts.
        """
        return None


# ---------------------------------------------------------------------------
# Mock simulator (deterministic, no rendering needed)
# ---------------------------------------------------------------------------
class MockSimulator(BaseSimulator):
    """Deterministic mock for CI / headless environments.

    Ignores real physics and simulates episode outcomes based on a
    heuristic: heavier corruptions have a lower baseline success rate
    plus small Gaussian noise.
    """

    def __init__(
        self,
        base_success_rate: float = 0.85,
        *,
        seed: int = 42,
    ) -> None:
        self.base_success_rate = base_success_rate
        self.rng = np.random.default_rng(seed)
        self._current_task: str | None = None

    def reset_task(self, task_id: int | str, seed: int = 0) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self._current_task = str(task_id)
        return np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        obs = np.zeros((128, 128, 3), dtype=np.uint8)
        return obs, 0.0, False, {}

    def run_episode(
        self,
        adapter: BaseWAMAdapter,
        task_id: int | str,
        max_steps: int = 100,
        seed: int = 0,
    ) -> EpisodeResult:
        _ = self.reset_task(task_id, seed=seed)
        # Deterministic heuristic: tasks with "heavy" in the name are
        # treated as harder.
        task_str = str(task_id)
        factor_str = getattr(adapter, "factor_name", task_str)
        heavy = 1.0 if "heavy" in factor_str.lower() else 0.0
        success = self.rng.random() < self.base_success_rate - heavy * 0.4
        steps = max_steps if not success else self.rng.integers(min(10, max_steps), max_steps if max_steps > 10 else max_steps + 1)
        return EpisodeResult(
            success=bool(success),
            steps=steps,
            total_reward=1.0 if success else 0.0,
            image=None,
            info={"mock": True, "task": task_str},
        )

    def list_tasks(self) -> list[str]:
        return ["mock_task_0", "mock_task_1", "mock_task_2"]


# ---------------------------------------------------------------------------
# LIBERO simulator
# ---------------------------------------------------------------------------
class LiberoSimulator(BaseSimulator):
    """Simulator wrapping the LIBERO benchmark (MuJoCo + robosuite).

    Lazy-imports ``libero`` so the package does not have to be installed
    unless this class is instantiated.

    Args:
        benchmark_name: One of ``libero_spatial``, ``libero_object``,
            ``libero_goal``, ``libero_90``, ``libero_10``, ``libero_100``.
        camera_name: Camera to extract observations from
            (default ``agentview``).
        render_gpu_device_id: ``-1`` for CPU/OSMesa, ``0`` for first GPU.
        image_size: (height, width) for rendered observations.
        max_steps: Hard step limit per episode.
    """

    _VALID_BENCHMARKS = {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_90",
        "libero_10",
        "libero_100",
    }

    def __init__(
        self,
        benchmark_name: str = "libero_spatial",
        *,
        camera_name: str = "agentview",
        render_gpu_device_id: int = -1,
        image_size: tuple[int, int] = (128, 128),
        max_steps: int = 100,
    ) -> None:
        if benchmark_name not in self._VALID_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark {benchmark_name!r}. "
                f"Choose from: {sorted(self._VALID_BENCHMARKS)}"
            )
        self.benchmark_name = benchmark_name
        self.camera_name = camera_name
        self.render_gpu_device_id = render_gpu_device_id
        self.image_size = image_size
        self.max_steps = max_steps

        # Lazily populated on first reset_task
        self._env: Any | None = None
        self._task_suite: Any | None = None
        self._current_task_id: int | None = None
        self._current_task_name: str | None = None
        self._current_task_bddl: str | None = None

        # Try to import libero only when this class is instantiated
        try:
            from libero.libero import benchmark, get_libero_path

            self._benchmark = benchmark
            self._get_libero_path = get_libero_path
            self._libero_available = True
        except ImportError as exc:
            raise ImportError(
                "libero is required for LiberoSimulator. "
                "Install with: pip install libero"
            ) from exc

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------
    def _build_suite(self) -> Any:
        """Load the task suite corresponding to ``benchmark_name``."""
        bm_dict = self._benchmark.get_benchmark_dict()
        return bm_dict[self.benchmark_name]()

    def list_tasks(self) -> list[str]:
        suite = self._build_suite()
        return [suite.get_task(i).name for i in range(suite.n_tasks)]

    def reset_task(self, task_id: int | str, seed: int = 0) -> np.ndarray:
        """Reset the simulator to a specific task and return the first frame."""
        if self._task_suite is None:
            self._task_suite = self._build_suite()

        if isinstance(task_id, str):
            # Resolve name to index
            names = self.list_tasks()
            if task_id not in names:
                raise ValueError(f"Task {task_id!r} not found in {self.benchmark_name}")
            task_id = names.index(task_id)

        task = self._task_suite.get_task(task_id)
        self._current_task_id = task_id
        self._current_task_name = task.name
        bddl = self._current_task_bddl = self._resolve_bddl(task)

        # Lazy env creation / recreation when task changes
        if self._env is not None:
            self._env.close()

        env_args: dict[str, Any] = {
            "bddl_file_name": bddl,
            "camera_heights": self.image_size[0],
            "camera_widths": self.image_size[1],
            "camera_names": [self.camera_name],
            "render_gpu_device_id": self.render_gpu_device_id,
            "has_offscreen_renderer": True,
        }

        from libero.libero.envs import OffScreenRenderEnv

        try:
            self._env = OffScreenRenderEnv(**env_args)
        except Exception as exc:
            warnings.warn(
                f"LIBERO off-screen rendering failed ({exc}). "
                "Falling back to MockSimulator. "
                "Install a renderer: apt install libosmesa6-dev || "
                "run under xvfb-run -a.",
                stacklevel=2,
            )
            raise _RenderingUnavailableError(
                "Off-screen rendering unavailable"
            ) from exc

        self._env.seed(seed)
        self._env.reset()
        init_states = self._task_suite.get_task_init_states(task_id)
        self._env.set_init_state(init_states[0])

        # In newer robosuite/libero, reset() returns obs directly
        # and _get_observation() was removed. Re-call to get fresh obs.
        obs = self._env.env.reset()
        return self._extract_image(obs)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Environment not created. Call reset_task() first.")
        obs, reward, done, info = self._env.step(action.tolist())
        img = self._extract_image(obs)
        return img, float(reward), bool(done), dict(info)

    def run_episode(
        self,
        adapter: BaseWAMAdapter,
        task_id: int | str,
        max_steps: int = 100,
        seed: int = 0,
    ) -> EpisodeResult:
        img = self.reset_task(task_id, seed=seed)
        total_reward = 0.0
        done = False
        steps = 0
        for _ in range(max_steps):
            if done:
                break
            action, _ = adapter.predict_action(img, state=self._current_task_name)
            action_arr = action if isinstance(action, np.ndarray) else action.detach().cpu().numpy()
            # Ensure action is the expected dimension (LIBERO: 7-DoF)
            if action_arr.ndim == 0:
                action_arr = np.zeros(7)
            elif action_arr.ndim == 1 and action_arr.shape[0] < 7:
                pad = np.zeros(7 - action_arr.shape[0])
                action_arr = np.concatenate([action_arr, pad])
            elif action_arr.shape[0] > 7:
                action_arr = action_arr[:7]

            img, reward, done, info = self.step(action_arr)
            total_reward += reward
            steps += 1

        success = bool(total_reward > 0.0 or info.get("success", False))
        return EpisodeResult(
            success=success,
            steps=steps,
            total_reward=total_reward,
            image=img,
            info=info,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_bddl(task: Any) -> str:
        """Resolve full BDDL file path from a task object."""
        import os

        from libero.libero import get_libero_path

        return os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )

    def _extract_image(self, obs: dict[str, Any]) -> np.ndarray:
        """Pull the camera image out of a robosuite observation dict."""
        key = f"{self.camera_name}_image"
        if key in obs:
            img = obs[key]
            # robosuite returns images as (H, W, 3) uint8 already
            return np.array(img, dtype=np.uint8)
        # Fallback: try without suffix
        if self.camera_name in obs:
            return np.array(obs[self.camera_name], dtype=np.uint8)
        raise KeyError(f"Camera image key {key!r} not found in observation. Keys: {list(obs.keys())}")

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class _RenderingUnavailableError(Exception):
    """Raised when LIBERO cannot create an off-screen render context."""
    pass
