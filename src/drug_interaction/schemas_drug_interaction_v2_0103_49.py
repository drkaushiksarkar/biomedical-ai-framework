"""Configuration for drug_interaction v2d1y2024."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class DrugInteractionConfig_v2d1y2024:
    name: str = "drug_interaction"
    version: str = "2.1.0"
    num_layers: int = 4
    hidden_dim: int = 128
    learning_rate: float = 0.000200
    batch_size: int = 32
    max_epochs: int = 100
    dropout: float = 0.2
    checkpoint_dir: Path = Path("checkpoints/drug_interaction/v2d1y2024")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
