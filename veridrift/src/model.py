from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class NodeGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        layer_norm_first: bool = True,
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.layer_norm_first = bool(layer_norm_first)
        self.use_ln = bool(use_ln)
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.lns.append(nn.LayerNorm(input_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_repr: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_states: Dict[str, torch.Tensor] = {}
        h = x
        if self.layer_norm_first:
            h = self.lns[0](h)
        for layer_idx, conv in enumerate(self.convs, start=1):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            if self.use_ln:
                h = self.lns[layer_idx](h)
            if layer_idx != len(self.convs):
                h = F.dropout(h, p=self.dropout, training=self.training)
            hidden_states[f"h{layer_idx}"] = h

        logits = self.classifier(hidden_states[f"h{len(self.convs)}"])
        if return_repr:
            return logits, hidden_states
        return logits
