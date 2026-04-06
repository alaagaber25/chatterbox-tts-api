from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from app.config import Config


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_t3_checkpoint(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()

    if candidate.is_file():
        return candidate

    preferred_files = [
        candidate / "t3_mtl23ls_v2.safetensors",
        candidate / "model.safetensors",
    ]
    for preferred in preferred_files:
        if preferred.exists():
            return preferred

    checkpoint_dirs = sorted(
        [p for p in candidate.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    if checkpoint_dirs:
        trainer_file = checkpoint_dirs[-1] / "model.safetensors"
        if trainer_file.exists():
            return trainer_file

    raise FileNotFoundError(f"Could not resolve a T3 checkpoint from: {candidate}")


def load_t3_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    resolved_path = resolve_t3_checkpoint(checkpoint_path)
    state_dict = load_file(str(resolved_path))
    if state_dict:
        first_key = next(iter(state_dict))
        if first_key.startswith("t3."):
            state_dict = {
                key.removeprefix("t3."): value for key, value in state_dict.items()
            }
    return state_dict


def resolve_configured_t3_checkpoint() -> Path:
    if not Config.TTS_CHECKPOINT_PATH:
        raise ValueError(
            "TTS_CHECKPOINT_PATH is not configured. "
            "Set TTS_CHECKPOINT_PATH to a T3 safetensors file or checkpoint directory."
        )
    return resolve_t3_checkpoint(Config.TTS_CHECKPOINT_PATH)
