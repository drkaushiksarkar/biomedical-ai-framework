"""Configuration for biomarker_selector v5d22y2024."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class BiomarkerSelectorConfig_v5d22y2024:
    name: str = "biomarker_selector"
    version: str = "5.22.0"
    num_layers: int = 10
    hidden_dim: int = 320
    learning_rate: float = 0.000500
    batch_size: int = 80
    max_epochs: int = 250
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/biomarker_selector/v5d22y2024")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
