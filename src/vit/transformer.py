import torch.nn as nn

from vit.attention import MultiHeadAttention
from vit.config import MODEL_SHAPE_DEBUG


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
        num_heads=4,
        mlp_dim=512
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):

        # attention branch
        attention_input = self.norm1(x)

        attention_output = self.attention(
            attention_input
        )

        x = x + attention_output

        if MODEL_SHAPE_DEBUG:
            print("after attention residual:", x.shape)

        # MLP branch
        mlp_input = self.norm2(x)

        mlp_output = self.mlp(
            mlp_input
        )

        x = x + mlp_output
        if MODEL_SHAPE_DEBUG:
            print("after MLP residual:", x.shape)

        return x
