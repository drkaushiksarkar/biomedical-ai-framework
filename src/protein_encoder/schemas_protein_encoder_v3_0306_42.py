"""Configuration for protein_encoder v3d28y2024."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class ProteinEncoderConfig_v3d28y2024:
    name: str = "protein_encoder"
    version: str = "3.28.0"
    num_layers: int = 6
    hidden_dim: int = 192
    learning_rate: float = 0.000300
    batch_size: int = 48
    max_epochs: int = 150
    dropout: float = 0.3
    checkpoint_dir: Path = Path("checkpoints/protein_encoder/v3d28y2024")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
