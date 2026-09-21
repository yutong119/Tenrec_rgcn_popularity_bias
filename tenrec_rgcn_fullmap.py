from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class TenRecFullGraphRGCN(nn.Module):
    """
    Pure FullGraph mul

    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ti-relational RGCN baseline.

    This model has NO popularity input, NO popularity penalty,
    NO popularity-aware reranking, and NO cross-domain transfer.
    Training is performed with BPR only in the experiment runner.

    Node indices:
        users: [0, num_users)
        items: [num_users, num_users + num_items)

    Relation IDs:
        0 click
        1 like
        2 share
        3 follow
        4 rev_click
        5 rev_like
        6 rev_share
        7 rev_follow
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_relations: int = 8,
        num_bases: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.num_relations = int(num_relations)

        self.user_embedding = nn.Embedding(
            self.num_users,
            self.embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            self.num_items,
            self.embedding_dim,
        )

        self.conv1 = RGCNConv(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            num_relations=self.num_relations,
            num_bases=num_bases,
        )

        self.conv2 = RGCNConv(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            num_relations=self.num_relations,
            num_bases=num_bases,
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(
            self.user_embedding.weight
        )
        nn.init.xavier_uniform_(
            self.item_embedding.weight
        )

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the whole graph ONCE.

        Returns:
            user_repr: [num_users, d]
            item_repr: [num_items, d]
        """
        x0 = torch.cat(
            [
                self.user_embedding.weight,
                self.item_embedding.weight,
            ],
            dim=0,
        )

        h1 = self.conv1(
            x0,
            edge_index,
            edge_type,
        )
        h1 = self.activation(h1)
        h1 = self.dropout(h1)

        h2 = self.conv2(
            h1,
            edge_index,
            edge_type,
        )

        # Residual path keeps direct trainable user/item ID embeddings.
        h = x0 + h2

        user_repr = h[: self.num_users]
        item_repr = h[self.num_users :]

        return user_repr, item_repr

    @staticmethod
    def score_pairs(
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = user_repr[user_idx]
        item_vectors = item_repr[item_idx]

        return (
            user_vectors
            * item_vectors
        ).sum(dim=-1)

    @staticmethod
    def score_user_batch_against_all_items(
        user_idx: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw dot-product scores:
            [batch_users, num_items]
        """
        user_vectors = user_repr[user_idx]
        return user_vectors @ item_repr.t()
