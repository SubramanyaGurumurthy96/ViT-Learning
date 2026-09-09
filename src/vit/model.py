import torch
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

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                (image_size//patch_size) * (image_size//patch_size) + 1,
                embedding_dim
            )
        )

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

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(
                64, 96,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(
                96, embedding_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

        # decoder
        out1 = embedding_dim - (embedding_dim//4)
        self.up1 = nn.ConvTranspose2d(
            embedding_dim, out1,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(192, out1, 3, padding=1),
            nn.BatchNorm2d(out1),
            nn.GELU(),

            nn.Conv2d(out1, out1, 3, padding=1),
            nn.BatchNorm2d(out1),
            nn.GELU(),
        )

        self.up2 = nn.ConvTranspose2d(
            out1, embedding_dim//2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim//2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim//2),
            nn.GELU(),

            nn.Conv2d(embedding_dim//2, embedding_dim//2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim//2),
            nn.GELU(),
        )

        self.up3 = nn.ConvTranspose2d(
            embedding_dim//2, embedding_dim//4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(embedding_dim//2, embedding_dim//4, 3, padding=1),
            nn.BatchNorm2d(embedding_dim//4),
            nn.GELU(),

            nn.Conv2d(embedding_dim//4, embedding_dim//4, 3, padding=1),
            nn.BatchNorm2d(embedding_dim//4),
            nn.GELU(),
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_dim//4, embedding_dim//8,
                kernel_size=2,
                stride=2
            ),
            nn.GELU(),
        )

        self.segmentation_head = nn.Conv2d(
            embedding_dim//8,
            num_segments,
            kernel_size=1
        )


    def forward(self, x):

        shape = x.shape

        #print("input: ", shape)
        # x = self.conv_stem(x)
        f80 = self.enc1(x)     # [B, 32, 80, 80]
        f40 = self.enc2(f80)   # [B, 64, 40, 40]
        f20 = self.enc3(f40)   # [B, 96, 20, 20]
        f10 = self.enc4(f20)   # [B,128, 10, 10]
        #print("conv stem output ",x.shape)
        
        x = f10.flatten(2)
        #print("conv stem flatten",x.shape)

        x = x.transpose(1, 2)
        #print("conv stem flatten + transpose: ", x.shape)

        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        x = x + self.position_embedding

        # x = self.patch_embedding(x)
        # print("after patch embedding: ", x.shape)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        #print(x.shape)

        cls_output = x[:, 0, :]
        #print("after cls: ", cls_output.shape)
        logits = self.classifier(cls_output)
        
        # segmentation branch
        #print("input to seg: ",x.shape)
        patch_tokens = x[:, 1:, :]
        # patch_tokens = x
        #print("patch token to seg ",patch_tokens.shape)
        patch_tokens = patch_tokens.transpose(1, 2)
        #print("patch token after transpose: ", patch_tokens.shape)
        patch_reshape = self.image_size // self.patch_size
        #print("grid size: ", patch_reshape)
        #print("patch size: ", self.patch_size)
        patch_tokens = patch_tokens.reshape(shape[0], self.embedding_dim, patch_reshape, patch_reshape)

        #########################
        d20 = self.up1(patch_tokens)    # [B,96,20,20]
        d20 = torch.cat(
            [d20, f20],
            dim = 1
        )
        d20 = self.dec1(d20)     # [B,96,20,20]

        #########################
        d40 = self.up2(d20)      # [B,64,40,40]

        d40 = torch.cat(
            [d40, f40],
            dim=1
        )

        d40 = self.dec2(d40)    # [B,64,40,40]

        #########################
        d80 = self.up3(d40)     # [B,32,80,80]
        d80 = torch.cat(
            [d80, f80],
            dim = 1
        )                       # [B,32,80,80]
        d80 = self.dec3(d80)

        #########################
        d160 = self.up4(d80)    # [B,16,160,160]
        seg_output = self.segmentation_head(d160)

        return logits, seg_output
