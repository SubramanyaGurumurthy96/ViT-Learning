import torch.nn as nn

from vit.patch_embedding import PatchEmbedding
from vit.transformer import TransformerBlock


class ViT(nn.Module):

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        embedding_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
        num_classes=10
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_output = x[:, 0, :]

        logits = self.classifier(cls_output)

        return logits
