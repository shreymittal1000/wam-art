"""Tests for episode-connected online WAM-ART scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from wam_art.eval.online import OnlineWAMARTScorer, fastwam_vae_latent_extractor


def _extract(image: torch.Tensor) -> torch.Tensor:
    return image.flatten().float()


def _image(value: float) -> torch.Tensor:
    image = torch.full((1, 3, 2, 2), value, dtype=torch.float32)
    image[..., 0, 0] += 0.25
    return image


def test_collect_reference_and_score_same_episode(tmp_path) -> None:
    reference_path = tmp_path / "nominal.npz"
    collector = OnlineWAMARTScorer(_extract, mode="collect", k=1)
    for value in (0.2, 0.3, 0.4, 0.5):
        collector.observe(_image(value))
    collector.end_episode(measured_success=True)
    collector.save_reference(reference_path)

    scorer = OnlineWAMARTScorer(
        _extract, mode="score", reference_path=reference_path, k=1
    )
    scorer.observe(_image(0.25))
    scorer.observe(torch.flip(_image(0.25), dims=(-1,)))
    scorer.end_episode(measured_success=False)
    report = scorer.build_report(
        model_name="test-wam",
        task_suite="libero_spatial",
        task_id=0,
        task_description="test task",
        corruption="occlusion",
        corruption_kwargs={"ratio": 0.35},
    )

    assert report.total_episodes == 1
    assert report.successes == 0
    assert report.measured_success_rate == 0.0
    assert report.episodes[0].measured_success is False
    assert report.episodes[0].predicted_success_rate is not None
    assert report.episodes[0].n_observations == 2

    output = tmp_path / "report.json"
    report.save(output)
    payload = json.loads(output.read_text())
    assert payload["episodes"][0]["measured_success"] is False
    assert payload["corruption_kwargs"] == {"ratio": 0.35}


def test_reference_uses_only_successful_episodes(tmp_path) -> None:
    collector = OnlineWAMARTScorer(_extract, mode="collect", k=1)
    for value in (0.1, 0.2, 0.3, 0.4):
        collector.observe(_image(value))
    collector.end_episode(True)
    collector.observe(torch.ones((1, 3, 2, 2)))
    collector.end_episode(False)
    path = collector.save_reference(tmp_path / "ref.npz")

    with np.load(path) as payload:
        # Four successful observations split 60/40 -> two reference latents.
        assert payload["reference_latents"].shape[0] == 2


def test_cannot_report_unfinished_or_unobserved_episode(tmp_path) -> None:
    collector = OnlineWAMARTScorer(_extract, mode="collect", k=1)
    with pytest.raises(RuntimeError, match="before any policy observations"):
        collector.end_episode(True)
    collector.observe(_image(0.2))
    with pytest.raises(RuntimeError, match="unfinished episode"):
        collector.build_report(
            model_name="x",
            task_suite="x",
            task_id=0,
            task_description="x",
            corruption=None,
        )


def test_score_requires_reference() -> None:
    with pytest.raises(ValueError, match="requires reference_path"):
        OnlineWAMARTScorer(_extract, mode="score")


def test_reference_k_must_match(tmp_path) -> None:
    path = tmp_path / "ref.npz"
    np.savez_compressed(
        path,
        reference_latents=np.ones((2, 3), dtype=np.float32),
        threshold=np.asarray(0.1),
        k=np.asarray(1),
    )
    with pytest.raises(ValueError, match="calibrated with k=1"):
        OnlineWAMARTScorer(_extract, mode="score", reference_path=path, k=2)


class _FakeFastWAM:
    def _encode_input_image_latents_tensor(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=0, keepdim=True)


def test_fastwam_extractor_uses_policy_tensor() -> None:
    extractor = fastwam_vae_latent_extractor(_FakeFastWAM())
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    latent = extractor(image)
    assert latent.shape == (4,)
    with pytest.raises(ValueError, match="shape"):
        extractor(image.squeeze(0))
