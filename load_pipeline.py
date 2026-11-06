"""
Bootstraps SpeechDiffusionPipeline entirely from the combined checkpoint.

Usage
-----
    from load_pipeline import load
    import torch

    pipe = load("combined_speech2image_checkpoint.pt")
    pipe = pipe.to("cuda")

    image = pipe("recording.wav").images[0]
    image.save("output.png")
"""

import sys
import types
import tempfile
import importlib
from pathlib import Path


def load(checkpoint_path: str, device: str | None = None):
    """
    Extract bundled source modules from the checkpoint and return
    a ready-to-use SpeechDiffusionPipeline without any local .py files.
    """
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Write bundled sources to a temp directory and add to sys.path
    _tmp = tempfile.mkdtemp(prefix="speech_pipeline_")
    for key, filename in [
        ("_model_src",          "model.py"),
        ("_load_centroids_src", "load_centroids.py"),
        ("_pipeline_src",       "speech_pipeline.py"),
    ]:
        Path(_tmp, filename).write_text(ckpt[key], encoding="utf-8")

    if _tmp not in sys.path:
        sys.path.insert(0, _tmp)

    # Force fresh import from the extracted sources
    for mod in ("model", "load_centroids", "speech_pipeline"):
        sys.modules.pop(mod, None)

    from speech_pipeline import SpeechDiffusionPipeline  # noqa: PLC0415
    return SpeechDiffusionPipeline.from_pretrained(checkpoint_path, device=device)
