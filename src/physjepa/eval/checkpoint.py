"""Load JEPA / pixel checkpoints and rebuild models from config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from physjepa.models import JEPAModule, PixelVideoModule

ModelType = Literal["jepa", "pixel"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_run_config(
    ckpt_path: str | Path,
    *,
    config_path: str | Path | None = None,
    model_type: ModelType | None = None,
) -> dict[str, Any]:
    ckpt_path = Path(ckpt_path)
    summary_path = ckpt_path.parent / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if isinstance(summary.get("config"), dict) and summary["config"]:
            return summary["config"]
    if config_path is not None:
        return load_yaml(config_path)
    root = Path(__file__).resolve().parents[3] / "configs"
    mt = model_type or infer_model_type(ckpt_path)
    name = "pixel_default.yaml" if mt == "pixel" else "jepa_default.yaml"
    default = root / name
    if default.exists():
        return load_yaml(default)
    raise FileNotFoundError(f"No config found for checkpoint {ckpt_path}")


def infer_model_type(ckpt_path: str | Path) -> ModelType:
    ckpt_path = Path(ckpt_path)
    if "pixel" in ckpt_path.parts:
        return "pixel"
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and raw.get("model_type") == "pixel":
            return "pixel"
    except Exception:
        pass
    return "jepa"


def build_jepa_from_config(cfg: dict[str, Any]) -> JEPAModule:
    m = cfg.get("model", cfg)
    return JEPAModule(
        latent_dim=int(m.get("latent_dim", 256)),
        base_channels=int(m.get("base_channels", 32)),
        context_len=int(m.get("context_len", 4)),
        pred_horizon=int(m.get("pred_horizon", 1)),
        predictor_hidden=int(m.get("predictor_hidden", 512)),
        predictor_layers=int(m.get("predictor_layers", 3)),
        ema_momentum=float(m.get("ema_momentum", 0.996)),
        predictor_type=str(m.get("predictor_type", "mlp")),
    )


def build_pixel_from_config(cfg: dict[str, Any]) -> PixelVideoModule:
    m = cfg.get("model", cfg)
    return PixelVideoModule(
        latent_dim=int(m.get("latent_dim", 256)),
        base_channels=int(m.get("base_channels", 32)),
        context_len=int(m.get("context_len", 4)),
        pred_horizon=int(m.get("pred_horizon", 1)),
        predictor_hidden=int(m.get("predictor_hidden", 512)),
        predictor_layers=int(m.get("predictor_layers", 3)),
        decoder_channels=int(m.get("decoder_channels", 256)),
    )


def load_jepa_checkpoint(
    ckpt_path: str | Path,
    *,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[JEPAModule, dict[str, Any]]:
    ckpt_path = Path(ckpt_path)
    cfg = resolve_run_config(ckpt_path, config_path=config_path, model_type="jepa")
    model = build_jepa_from_config(cfg)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_state_dict(state)
    device = torch.device(device)
    model.to(device)
    model.eval()
    return model, cfg


def load_pixel_checkpoint(
    ckpt_path: str | Path,
    *,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[PixelVideoModule, dict[str, Any]]:
    ckpt_path = Path(ckpt_path)
    cfg = resolve_run_config(ckpt_path, config_path=config_path, model_type="pixel")
    model = build_pixel_from_config(cfg)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_state_dict(state)
    device = torch.device(device)
    model.to(device)
    model.eval()
    return model, cfg


def load_model_checkpoint(
    ckpt_path: str | Path,
    *,
    config_path: str | Path | None = None,
    model_type: ModelType | None = None,
    device: str | torch.device = "cpu",
) -> tuple[JEPAModule | PixelVideoModule, dict[str, Any], ModelType]:
    mt = model_type or infer_model_type(ckpt_path)
    if mt == "pixel":
        model, cfg = load_pixel_checkpoint(ckpt_path, config_path=config_path, device=device)
    else:
        model, cfg = load_jepa_checkpoint(ckpt_path, config_path=config_path, device=device)
    return model, cfg, mt


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
