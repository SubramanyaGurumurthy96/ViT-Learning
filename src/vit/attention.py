import math

import torch.nn as nn
import torch.nn.functional as F

from vit.config import MODEL_SHAPE_DEBUG


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim=128, num_heads=4):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.last_attention = None

    def forward(self, x):

        B, N, D = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        if MODEL_SHAPE_DEBUG:
            print("Q before split:", Q.shape)

        # [B, N, 128]
        # ->
        # [B, N, 4, 32]

        Q = Q.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim
        )

        K = K.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim
        )

        V = V.reshape(
            B,
            N,
            self.num_heads,
            self.head_dim
        )

        # [B, N, H, D_head]
        # ->
        # [B, H, N, D_head]

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        if MODEL_SHAPE_DEBUG:
            print("Q after split:", Q.shape)

        K_T = K.transpose(-2, -1)

        scores = Q @ K_T
        if MODEL_SHAPE_DEBUG:
            print("scores:", scores.shape)

        scores = scores / math.sqrt(self.head_dim)

        attention = F.softmax(
            scores,
            dim=-1
        )

        self.last_attention = attention.detach()

        output = attention @ V
        if MODEL_SHAPE_DEBUG:
            print("per-head output:", output.shape)

        # [B, H, N, head_dim]
        # ->
        # [B, N, H, head_dim]

        output = output.permute(
            0, 2, 1, 3
        )

        # combine 4 × 32 back into 128

        output = output.reshape(
            B,
            N,
            self.embedding_dim
        )

        if MODEL_SHAPE_DEBUG:
            print("combined:", output.shape)

        output = self.output_projection(output)

        return output
