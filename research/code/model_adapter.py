"""Hash-verified import adapter for the unchanged conference DeepAA model."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from .constants import (
    BASELINE_SOURCE,
    BASELINE_SOURCE_SHA256,
    IMAGE_SIZE,
    LOSS_WEIGHTS,
)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline_module(repository: Path):
    source = repository.resolve() / BASELINE_SOURCE
    actual = source_sha256(source)
    if actual != BASELINE_SOURCE_SHA256:
        raise RuntimeError(
            f"Protected conference source hash mismatch: expected={BASELINE_SOURCE_SHA256}; actual={actual}"
        )
    module_name = "protected_data2026_baseline"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import protected baseline from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_model(repository: Path, k: int):
    if k < 2:
        raise ValueError("K must be at least 2")
    baseline = load_baseline_module(repository)
    cfg = baseline.DAAConfig(
        img_size=IMAGE_SIZE,
        latent_dim=k - 1,
        num_classes=7,
        recon_w=LOSS_WEIGHTS["reconstruction"],
        class_w=LOSS_WEIGHTS["classification"],
        kl_w=LOSS_WEIGHTS["kl"],
        at_w=LOSS_WEIGHTS["archetype_simplex"],
    )
    model = baseline.DeepAA(cfg)
    return baseline, model, cfg


def provenance() -> dict:
    return {
        "source": BASELINE_SOURCE,
        "source_sha256": BASELINE_SOURCE_SHA256,
        "reused_symbols": [
            "DAAConfig",
            "DeepAA",
            "create_z_fixed",
            "train_step",
            "eval_step",
            "kl_standard_normal",
        ],
        "copied_model_logic": False,
        "note": "The journal runner imports the protected model/loss implementation; only manifest/determinism/output infrastructure is separate.",
    }
