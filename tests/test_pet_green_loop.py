from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "pet_green_loop.py"


def load_module(output_dir: Path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(output_dir)
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("pet_green_loop", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_frames() -> torch.Tensor:
    frames = np.zeros((5, 64, 64, 3), dtype=np.float32)
    frames[..., 1] = 1.0
    for index in range(5):
        frames[index, 16:48, 20 + index : 44 + index] = (0.80, 0.20, 0.10)
    return torch.from_numpy(frames)


def test_keyer_exports_closed_animated_webp(tmp_path):
    module = load_module(tmp_path)
    result = module.PetChromaKeyClosedLoopWebP().key_and_save(
        sample_frames(), "test/loop", 24.0, "#00FF00", 0.20, 0.10, 0.85, True, True, 100
    )
    preview, mask, saved = result["result"]
    assert preview.shape == (5, 64, 64, 3)
    assert mask.shape == (5, 64, 64)
    assert torch.equal(mask[0], mask[-1])
    assert float(mask[0, 0, 0]) == 0.0
    assert float(mask[0, 32, 32]) > 0.99
    webp = Image.open(saved)
    assert webp.n_frames == 5


def test_normalizer_preserves_interior_subject(tmp_path):
    module = load_module(tmp_path)
    source = sample_frames()[:1]
    normalized = module.PetNormalizeGreenFirstFrame().normalize(source, "#00FF00", 0.30, 1)[0]
    assert torch.allclose(normalized[0, 0, 0], torch.tensor([0.0, 1.0, 0.0]))
    expected = torch.tensor([204, 51, 25], dtype=torch.float32) / 255.0
    assert torch.allclose(normalized[0, 32, 32], expected)
