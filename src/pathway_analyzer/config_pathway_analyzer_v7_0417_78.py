"""Configuration for pathway_analyzer v7d46y2024."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class PathwayAnalyzerConfig_v7d46y2024:
    name: str = "pathway_analyzer"
    version: str = "7.46.0"
    num_layers: int = 14
    hidden_dim: int = 448
    learning_rate: float = 0.000700
    batch_size: int = 112
    max_epochs: int = 350
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/pathway_analyzer/v7d46y2024")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
