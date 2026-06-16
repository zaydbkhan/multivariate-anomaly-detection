"""
SynCAN Model Registry

Manages a single global TranAD model for the SynCAN dataset, including
checkpoint save/load and scorer state (thresholds).

Filesystem layout:
    models/syncan/
        model.ckpt          -- PyTorch checkpoint (from training)
        scorer_state.json   -- calibrated thresholds + POT params (from evaluation)
        eval_results.json   -- evaluation metrics per attack type
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

import src.model
from src.model import TranADConfig, TranADNet

sys.modules["tranad_model"] = src.model
sys.modules["app.tranad_model"] = src.model

logger = logging.getLogger(__name__)


class SynCANRegistry:
    """Registry managing the single global SynCAN model.

    Args:
        base_dir: Root directory for model artifacts (default: models/syncan).
    """

    def __init__(self, base_dir: str | Path = "models/syncan"):
        self.base_dir = Path(base_dir)
        self._model_cache: tuple[TranADNet, TranADConfig] | None = None

    @property
    def ckpt_path(self) -> Path:
        return self.base_dir / "model.ckpt"

    @property
    def scorer_path(self) -> Path:
        return self.base_dir / "scorer_state.json"

    @property
    def eval_path(self) -> Path:
        return self.base_dir / "eval_results.json"

    def get_model(
        self, device: str | torch.device = "cpu"
    ) -> tuple[TranADNet, TranADConfig]:
        """Load the trained SynCAN model.

        Returns cached model if already loaded.

        Raises:
            FileNotFoundError: If no checkpoint exists.
        """
        device = torch.device(device) if isinstance(device, str) else device

        if self._model_cache is not None:
            model, config = self._model_cache
            return model.to(device), config

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.ckpt_path}")

        checkpoint = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = TranADNet(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self._model_cache = (model, config)
        logger.info("Loaded SynCAN model on %s", device)
        return model, config

    def save_model(
        self, model: TranADNet, config: TranADConfig, final_loss: float, epoch: int = 0
    ) -> Path:
        """Save model checkpoint."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "final_loss": final_loss,
            },
            self.ckpt_path,
        )
        logger.info("Model saved to %s", self.ckpt_path)
        return self.ckpt_path

    def save_scorer_state(self, state: dict) -> Path:
        """Save scoring state (thresholds, method, params)."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                serializable[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                serializable[k] = int(v)
            elif isinstance(v, dict):
                serializable[k] = {
                    sk: float(sv) if isinstance(sv, (np.floating, float)) else sv
                    for sk, sv in v.items()
                }
            else:
                serializable[k] = v
        with open(self.scorer_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Scorer state saved to %s", self.scorer_path)
        return self.scorer_path

    def get_scorer_state(self) -> dict | None:
        """Load saved scoring state, or None if not saved."""
        if not self.scorer_path.exists():
            return None
        with open(self.scorer_path) as f:
            return json.load(f)

    def save_eval_results(self, results: dict) -> Path:
        """Save evaluation results."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, float)):
                serializable[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                serializable[k] = int(v)
            elif isinstance(v, dict):
                serializable[k] = {
                    sk: float(sv) if isinstance(sv, (np.floating, float)) else sv
                    for sk, sv in v.items()
                }
            else:
                serializable[k] = v
        with open(self.eval_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Eval results saved to %s", self.eval_path)
        return self.eval_path

    def clear_cache(self) -> None:
        self._model_cache = None
