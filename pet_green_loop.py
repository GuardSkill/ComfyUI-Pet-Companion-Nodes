"""Production-oriented green-screen helpers for pet companion loops."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
import folder_paths
import numpy as np
import torch
from PIL import Image


def _rgb8(images: torch.Tensor) -> np.ndarray:
    data = images.detach().cpu().numpy()
    if data.ndim != 4 or data.shape[-1] < 3:
        raise ValueError("images must have shape [frames, height, width, channels]")
    return np.clip(data[..., :3] * 255.0, 0, 255).astype(np.uint8)


def _parse_hex(value: str) -> np.ndarray:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", value.strip())
    if not match:
        raise ValueError("key_color must be a six-digit RGB hex value, for example #00FF00")
    raw = match.group(1)
    return np.asarray([int(raw[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _edge_connected(candidate: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(candidate, dtype=bool)
    edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    edge_labels = edge_labels[edge_labels != 0]
    return np.isin(labels, edge_labels)


def _alpha_and_despill(
    frame: np.ndarray,
    key: np.ndarray,
    similarity: float,
    smoothness: float,
    despill: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = frame.astype(np.float32) / 255.0
    key01 = key / 255.0
    distance = np.sqrt(np.mean(np.square(rgb - key01), axis=2))
    outer = min(1.0, similarity + smoothness)
    connected = _edge_connected(distance <= outer)
    t = np.clip((distance - similarity) / max(smoothness, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    alpha = np.where(connected, smooth, 1.0).astype(np.float32)

    # Suppress green spill only near the keyed boundary, leaving interior coat colors intact.
    output = rgb.copy()
    edge_weight = np.clip((1.0 - alpha) + cv2.GaussianBlur(1.0 - alpha, (0, 0), 1.25), 0.0, 1.0)
    rb_max = np.maximum(output[..., 0], output[..., 2])
    excess = np.maximum(output[..., 1] - rb_max, 0.0)
    output[..., 1] -= excess * edge_weight * despill
    return np.clip(output, 0.0, 1.0), alpha


def _durations(frame_count: int, fps: float) -> list[int]:
    boundaries = [round(index * 1000.0 / fps) for index in range(frame_count + 1)]
    return [max(1, boundaries[index + 1] - boundaries[index]) for index in range(frame_count)]


def _safe_prefix(value: str) -> tuple[Path, str]:
    normalized = value.replace("\\", "/").strip("/") or "pet_companion/loop"
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", part) for part in normalized.split("/") if part]
    if not parts:
        parts = ["pet_companion", "loop"]
    return Path(*parts[:-1]), parts[-1]


class PetNormalizeGreenFirstFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "key_color": ("STRING", {"default": "#00FF00"}),
                "background_tolerance": ("FLOAT", {"default": 0.30, "min": 0.02, "max": 0.75, "step": 0.01}),
                "edge_feather": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("exact_green_image",)
    FUNCTION = "normalize"
    CATEGORY = "Pet Companion/Green Screen"

    def normalize(self, image, key_color, background_tolerance, edge_feather):
        frames = _rgb8(image)
        key = _parse_hex(key_color)
        result = []
        for frame in frames:
            rgb = frame.astype(np.float32)
            border = np.concatenate((rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3), rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)))
            inferred = np.median(border, axis=0)
            distance = np.sqrt(np.mean(np.square((rgb - inferred) / 255.0), axis=2))
            connected = _edge_connected(distance <= background_tolerance)
            if edge_feather:
                connected = cv2.dilate(connected.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=edge_feather).astype(bool)
            output = frame.copy()
            output[connected] = key.astype(np.uint8)
            result.append(output.astype(np.float32) / 255.0)
        return (torch.from_numpy(np.stack(result)),)


class PetChromaKeyClosedLoopWebP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "pet_companion/loop"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "key_color": ("STRING", {"default": "#00FF00"}),
                "similarity": ("FLOAT", {"default": 0.20, "min": 0.01, "max": 0.70, "step": 0.01}),
                "smoothness": ("FLOAT", {"default": 0.10, "min": 0.001, "max": 0.40, "step": 0.005}),
                "despill": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "close_loop": ("BOOLEAN", {"default": True}),
                "lossless": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("checker_preview", "foreground_mask", "saved_webp")
    FUNCTION = "key_and_save"
    CATEGORY = "Pet Companion/Green Screen"
    OUTPUT_NODE = True

    def key_and_save(self, images, filename_prefix, fps, key_color, similarity, smoothness, despill, close_loop, lossless, quality):
        frames = _rgb8(images)
        if len(frames) < 2:
            raise ValueError("At least two video frames are required")
        key = _parse_hex(key_color)
        rgba_frames = []
        previews = []
        masks = []
        for index, frame in enumerate(frames):
            rgb, alpha = _alpha_and_despill(frame, key, similarity, smoothness, despill)
            rgba = np.dstack((np.round(rgb * 255.0).astype(np.uint8), np.round(alpha * 255.0).astype(np.uint8)))
            rgba_frames.append(rgba)
            masks.append(alpha)
            yy, xx = np.indices(alpha.shape)
            checker = np.where(((xx // 16 + yy // 16) % 2)[..., None] == 0, 0.18, 0.32).astype(np.float32)
            previews.append(rgb * alpha[..., None] + checker * (1.0 - alpha[..., None]))

        if close_loop:
            rgba_frames[-1] = rgba_frames[0].copy()
            previews[-1] = previews[0].copy()
            masks[-1] = masks[0].copy()

        subfolder, stem = _safe_prefix(filename_prefix)
        output_root = Path(folder_paths.get_output_directory())
        target_dir = output_root / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{stem}_{stamp}.webp"
        target = target_dir / filename
        pil = [Image.fromarray(frame) for frame in rgba_frames]
        pil[0].save(
            target,
            save_all=True,
            append_images=pil[1:],
            duration=_durations(len(pil), fps),
            loop=0,
            lossless=lossless,
            quality=quality,
            method=6,
            minimize_size=False,
        )

        preview_tensor = torch.from_numpy(np.stack(previews).astype(np.float32))
        mask_tensor = torch.from_numpy(np.stack(masks).astype(np.float32))
        ui_item = {"filename": filename, "subfolder": subfolder.as_posix() if str(subfolder) != "." else "", "type": "output", "animated": True}
        return {"ui": {"images": [ui_item]}, "result": (preview_tensor, mask_tensor, str(target))}


NODE_CLASS_MAPPINGS = {
    "PetNormalizeGreenFirstFrame": PetNormalizeGreenFirstFrame,
    "PetChromaKeyClosedLoopWebP": PetChromaKeyClosedLoopWebP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PetNormalizeGreenFirstFrame": "Pet / Normalize Green First Frame",
    "PetChromaKeyClosedLoopWebP": "Pet / Chroma Key + Closed Loop WebP",
}
