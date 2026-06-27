"""Tests for VLM-guided perturbation editor."""

import numpy as np
import pytest

from wam_art.editing.vlm_editor import GeminiPerturbationEditor, VLMPerturbationEditor


class TestVLMPerturbationEditorInternals:
    def test_encode_image(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        b64 = VLMPerturbationEditor._encode_image(img)
        assert b64.startswith("data:image/png;base64,")
        assert len(b64) > 100

    def test_extract_json_clean(self):
        text = '{"corruption": "motion_blur", "params": {"kernel_size": 5}}'
        parsed = VLMPerturbationEditor._extract_json(text)
        assert parsed["corruption"] == "motion_blur"

    def test_extract_json_markdown(self):
        text = '```json\n{"corruption": "gaussian_blur", "params": {"sigma": 1.0}}\n```'
        parsed = VLMPerturbationEditor._extract_json(text)
        assert parsed["corruption"] == "gaussian_blur"

    def test_extract_json_with_prose(self):
        text = 'Some explanation then {"corruption": "occlusion", "params": {"ratio": 0.2}} and more text'
        parsed = VLMPerturbationEditor._extract_json(text)
        assert parsed["corruption"] == "occlusion"

    def test_parse_planner_response(self):
        text = '{"corruption": "brightness_shift", "params": {"factor": 0.6}, "explanation": "dim lighting"}'
        corr, params, exp = VLMPerturbationEditor._parse_planner_response(text)
        assert corr == "brightness_shift"
        assert params == {"factor": 0.6}
        assert exp == "dim lighting"

    def test_parse_planner_response_invalid(self):
        with pytest.raises(ValueError, match="Expected 'corruption' string"):
            VLMPerturbationEditor._parse_planner_response('{"params": {}}')


class TestGeminiPerturbationEditor:
    def test_default_model(self):
        editor = GeminiPerturbationEditor(
            factor_name="test",
            api_key="fake_key_for_test",
        )
        assert editor.model == "openai/gpt-4o"

    def test_missing_api_key(self):
        # Temporarily clear env var if present
        import os
        old_key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="requires an API key"):
                GeminiPerturbationEditor(factor_name="test")
        finally:
            if old_key is not None:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_fail_open_on_api_error(self, monkeypatch):
        editor = GeminiPerturbationEditor(
            factor_name="test",
            api_key="fake_key",
        )

        def fake_call(*args, **kwargs):
            raise RuntimeError("API down")

        monkeypatch.setattr(editor, "_call_api", fake_call)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="planning failed"):
            editor.edit(img, "make it blurry")

    def test_fail_open_on_unknown_corruption(self, monkeypatch):
        editor = GeminiPerturbationEditor(
            factor_name="test",
            api_key="fake_key",
        )

        def fake_call(*args, **kwargs):
            return {
                "choices": [
                    {"message": {"content": '{"corruption": "nonexistent", "params": {}}'}}
                ]
            }

        monkeypatch.setattr(editor, "_call_api", fake_call)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out = editor.edit(img, "do something weird")
        np.testing.assert_array_equal(out, img)
