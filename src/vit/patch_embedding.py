import torch
import torch.nn as nn

from vit.config import MODEL_SHAPE_DEBUG


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        channels=3,
        embedding_dim=128
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # 32 / 4 = 8 patches along each direction
        # 8 * 8 = 64 patches
        self.num_patches = (image_size // patch_size) ** 2

        # 4 * 4 * 3 = 48
        self.patch_dim = patch_size * patch_size * channels

        # 48 -> 128
        self.projection = nn.Linear(
            self.patch_dim,
            embedding_dim
        )

        # One learnable CLS token
        # shape: (1, 1, 128)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

        # 64 patches + 1 CLS = 65 positions
        # shape: (1, 65, 128)
        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_patches + 1,
                embedding_dim
            )
        )

    def forward(self, images):

        B, C, H, W = images.shape

        # ------------------------------------------------
        # 1. Create patches
        # ------------------------------------------------

        patches = images.unfold(
            2,
            self.patch_size,
            self.patch_size
        )

        patches = patches.unfold(
            3,
            self.patch_size,
            self.patch_size
        )

        # Currently:
        #
        # (B, C, 8, 8, 4, 4)

        # Move patch grid before channel/pixel dimensions
        patches = patches.permute(
            0, 2, 3, 1, 4, 5
        )

        # (B, 8, 8, 3, 4, 4)
        patches = patches.reshape(
            B,
            self.num_patches,
            self.patch_dim
        )

        # (B, 64, 48)
        if MODEL_SHAPE_DEBUG:
            print("patches:", patches.shape)

        # ------------------------------------------------
        # 2. Patch embedding
        # ------------------------------------------------

        x = self.projection(patches)

        # (B, 64, 128)
        if MODEL_SHAPE_DEBUG:
            print("patch embeddings:", x.shape)

        # ------------------------------------------------
        # 3. Add CLS token
        # ------------------------------------------------

        cls = self.cls_token.expand(
            B,
            -1,
            -1
        )

        # cls:
        # (B, 1, 128)

        x = torch.cat(
            [cls, x],
            dim=1
        )

        # (B, 65, 128)
        if MODEL_SHAPE_DEBUG:
            print("after CLS:", x.shape)

        # ------------------------------------------------
        # 4. Add positional embedding
        # ------------------------------------------------

        x = x + self.pos_embedding

        # still:
        # (B, 65, 128)
        if MODEL_SHAPE_DEBUG:
            print("after position:", x.shape)
        return x
