import torch.nn as nn
import torch.nn.functional as F
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
        num_classes=10,
        num_segments=3,
        kernal_size=1,
    ):
        super().__init__()

        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size

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

        self.segmentation_head = nn.Conv2d(
            embedding_dim,
            num_segments,
            kernel_size=kernal_size
        )



    def forward(self, x):

        shape = x.shape
        x = self.patch_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        # segmentation branch
        # print(x.shape)
        patch_tokens = x[:, 1:, :]
        # print(patch_tokens.shape)
        patch_tokens = patch_tokens.transpose(1, 2)
        # print(patch_tokens.shape)
        patch_reshape = self.image_size // self.patch_size
        patch_tokens = patch_tokens.reshape(shape[0], self.embedding_dim, patch_reshape, patch_reshape)

        seg_output = self.segmentation_head(patch_tokens)

        seg_output = F.interpolate(
            seg_output,
            size=(self.image_size, self.image_size),
            mode = 'bilinear',
            align_corners = False
        )

        return logits, seg_output
