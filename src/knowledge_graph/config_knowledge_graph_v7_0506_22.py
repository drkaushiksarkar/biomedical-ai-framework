"""Configuration for knowledge_graph v7d52y2024."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class KnowledgeGraphConfig_v7d52y2024:
    name: str = "knowledge_graph"
    version: str = "7.52.0"
    num_layers: int = 14
    hidden_dim: int = 448
    learning_rate: float = 0.000700
    batch_size: int = 112
    max_epochs: int = 350
    dropout: float = 0.5
    checkpoint_dir: Path = Path("checkpoints/knowledge_graph/v7d52y2024")
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])

    def validate(self) -> bool:
        assert self.num_layers > 0
        assert self.hidden_dim > 0
        return True
