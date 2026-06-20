"""Arquitetura MLP com PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron configurável.

    Args:
        input_dim: Número de features de entrada.
        output_dim: Número de classes de saída.
        hidden_dims: Lista com o tamanho de cada camada oculta.
        dropout: Taxa de dropout aplicada após cada camada oculta.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        """Total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
