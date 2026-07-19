"""Tests for the episode-connected RoboTwin integration helper."""

from __future__ import annotations

import json

import numpy as np
import torch

from wam_art.eval.robotwin import RobotTwinWAMARTSession


class _FakeFastWAM:
    def _encode_input_image_latents_tensor(self, image: torch.Tensor) -> torch.Tensor:
        return image.float().flatten()


def _tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float()


def test_seeded_corruption_is_repeatable_and_metadata_is_saved(tmp_path) -> None:
    reference = tmp_path / "reference.npz"
    np.savez_compressed(
        reference,
        reference_latents=np.eye(4, 12, dtype=np.float32),
        threshold=np.asarray(0.5),
        k=np.asarray(1),
    )
    output = tmp_path / "report.json"
    image = np.full((2, 2, 3), 127, dtype=np.uint8)

    def make_session() -> RobotTwinWAMARTSession:
        return RobotTwinWAMARTSession(
            _FakeFastWAM(),
            mode="score",
            checkpoint_path="robotwin.pt",
            task_name="click_alarmclock",
            output_path=output,
            reference_path=reference,
            corruption="gaussian_noise",
            corruption_kwargs={"sigma": 0.05},
            policy_seed=42,
            corruption_seed=7,
            k=1,
        )

    first = make_session()
    second = make_session()
    first_image = first.transform(image)
    second_image = second.transform(image)
    assert np.array_equal(first_image, second_image)

    first.observe(_tensor(first_image))
    first.end_episode(False, environment_seed=100042, policy_seed=43)

    payload = json.loads(output.read_text())
    assert payload["task_suite"] == "robotwin"
    assert payload["episodes"][0]["metadata"] == {
        "environment_seed": 100042,
        "policy_seed": 43,
        "corruption_seed": 7,
        "corruption_seed_derivation": (
            "base + episode_index * 1000000 + observation_index"
        ),
    }
