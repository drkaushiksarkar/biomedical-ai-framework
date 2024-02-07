"""Configuration for adverse_event v2d17y2024."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class AdverseEventConfig_v2d17y2024:
    name: str = "adverse_event"
    version: str = "2.17.0"
    num_layers: int = 4
    hidden_dim: int = 128
    learning_rate: float = 0.000200
    batch_size: int = 32
    max_epochs: int = 100
    dropout: float = 0.2
    checkpoint_dir: Path = Path("checkpoints/adverse_event/v2d17y2024")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
