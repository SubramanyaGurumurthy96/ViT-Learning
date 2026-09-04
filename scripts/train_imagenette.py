import argparse
from itertools import accumulate
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vit import ViT
from vit.data import load_imagenette
from vit.utils import get_gradient_norm, validate


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def log_heatmap(writer, tag, data_2d, title, step):
    fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(
        data_2d
        .detach()
        .cpu()
        .numpy()
    )

    ax.set_title(title)

    fig.colorbar(im)

    writer.add_figure(tag, fig, step)

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT on Imagenette")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(
            PROJECT_ROOT, "imageNette", "imagenette2-160"
        ),
        help="Directory containing Imagenette train/ and val/ folders",
    )
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-path",
        default=os.path.join(
            SCRIPT_DIR, "checkpoints", "best_vit_imagenette.pth"
        ),
    )
    parser.add_argument(
        "--runs-dir",
        default=os.path.join(PROJECT_ROOT, "runs", "vit_imagenette"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(
        os.path.dirname(args.checkpoint_path) or ".",
        exist_ok=True
    )

    checkpoint_path = args.checkpoint_path

    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    train_dataset, val_dataset = load_imagenette(
        args.data_dir,
        image_size=args.image_size
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    model = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embedding_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
        num_classes=10
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print("Checkpoint loaded")

    criterion = nn.CrossEntropyLoss()

    print_once = False

    global_step = 0
    writer = SummaryWriter(args.runs_dir)

    attention_grid = args.image_size // args.patch_size

    # -----------------------------------------
    # Capture patch-embedding layer output for
    # visualization via a forward hook. Stores
    # the tensor of shape [B, num_patches + 1, embedding_dim].
    # -----------------------------------------

    patch_activation = {}

    def capture_patch_embedding(module, inputs, output):
        patch_activation["output"] = output.detach()

    model.patch_embedding.register_forward_hook(
        capture_patch_embedding
    )

    accumulatation_step = 2

    for epoch in range(args.epochs):

        # =====================================================
        # TRAINING
        # =====================================================

        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for batch_idx, (batch_images, batch_labels) in enumerate(loader):

            # -----------------------------------------
            # Prepare input
            # -----------------------------------------

            batch_images = batch_images.float()

            # -----------------------------------------
            # Forward pass
            # -----------------------------------------

            output = model(batch_images)

            predictions = output.argmax(dim=1)

            # -----------------------------------------
            # Print shapes once
            # -----------------------------------------

            if not print_once:

                print("batch images:", batch_images.shape)
                print("batch labels:", batch_labels.shape)
                print("output:", output.shape)

                print_once = True

            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = criterion(
                output,
                batch_labels
            )

            loss = loss / accumulatation_step

            # -----------------------------------------
            # Accuracy
            # -----------------------------------------

            accuracy = (
                predictions == batch_labels
            ).float().mean()

            # Accumulate epoch statistics

            train_loss_sum += (
                loss.item() * batch_images.size(0)
            )

            train_correct += (
                predictions == batch_labels
            ).sum().item()

            train_samples += batch_images.size(0)

            # -----------------------------------------
            # TensorBoard training
            # -----------------------------------------

            writer.add_scalar(
                "training/loss",
                loss.item(),
                global_step
            )

            writer.add_scalar(
                "training/accuracy",
                accuracy.item(),
                global_step
            )

            # -----------------------------------------
            # Layer heatmap visualization
            # -----------------------------------------

            if global_step % 5 == 0:

                # -----------------------------------------
                # Patch embedding layer
                #
                # patch_activation["output"]:
                # [B, num_patches + 1, embedding_dim]
                # -----------------------------------------

                patch_output = patch_activation.get("output")

                if patch_output is not None:

                    # First image, drop the CLS token
                    # [num_patches, embedding_dim]
                    patch_tokens = patch_output[0, 1:, :]

                    # Per-patch embedding magnitude,
                    # laid back out on the patch grid
                    patch_norms = patch_tokens.norm(
                        dim=1
                    ).reshape(
                        attention_grid,
                        attention_grid
                    )

                    log_heatmap(
                        writer,
                        "patch_embedding/token_norms",
                        patch_norms,
                        f"Epoch {epoch} - Patch Norms",
                        global_step
                    )

                    # Full token-by-dimension embedding map
                    # [num_patches, embedding_dim]
                    log_heatmap(
                        writer,
                        "patch_embedding/embeddings",
                        patch_tokens,
                        f"Epoch {epoch} - Patch Embeddings",
                        global_step
                    )

                # -----------------------------------------
                # Attention blocks (every block, every head)
                #
                # block.attention.last_attention:
                # [B, num_heads, N, N]
                # -----------------------------------------

                for block_idx, block in enumerate(model.blocks):

                    attention = (
                        block.attention.last_attention
                    )

                    num_heads = attention.shape[1]

                    # First image, CLS -> image patches
                    for head_idx in range(num_heads):

                        cls_attention = attention[
                            0,
                            head_idx,
                            0,
                            1:
                        ]

                        heatmap = cls_attention.reshape(
                            attention_grid,
                            attention_grid
                        )

                        log_heatmap(
                            writer,
                            f"attention/"
                            f"block_{block_idx}_head_{head_idx}",
                            heatmap,
                            f"Epoch {epoch} "
                            f"- Block {block_idx} "
                            f"- Head {head_idx}",
                            global_step
                        )

                    # Head-averaged CLS attention
                    mean_attention = attention[
                        0,
                        :,
                        0,
                        1:
                    ].mean(dim=0).reshape(
                        attention_grid,
                        attention_grid
                    )

                    log_heatmap(
                        writer,
                        f"attention/block_{block_idx}_mean",
                        mean_attention,
                        f"Epoch {epoch} "
                        f"- Block {block_idx} "
                        f"- Head Mean",
                        global_step
                    )

            # -----------------------------------------
            # Backpropagation
            # -----------------------------------------

            loss.backward()

            # -----------------------------------------
            # Gradient visualization
            # -----------------------------------------

            total_grad_norm = get_gradient_norm(
                model.parameters()
            )

            writer.add_scalar(
                "gradients/total",
                total_grad_norm,
                global_step
            )

            writer.add_scalar(
                "gradients/patch_embedding",
                get_gradient_norm(
                    model.patch_embedding.parameters()
                ),
                global_step
            )

            for block_idx, block in enumerate(model.blocks):

                writer.add_scalar(
                    f"gradients/block_{block_idx}",
                    get_gradient_norm(
                        block.parameters()
                    ),
                    global_step
                )

            writer.add_scalar(
                "gradients/classifier",
                get_gradient_norm(
                    model.classifier.parameters()
                ),
                global_step
            )

            # -----------------------------------------
            # Update parameters
            # -----------------------------------------
            if (batch_idx + 1) %accumulatation_step == 0:
                optimizer.step()

                # -----------------------------------------
                # Clear previous gradients
                # -----------------------------------------

                optimizer.zero_grad(set_to_none=True)

            # -----------------------------------------
            # Terminal output
            # -----------------------------------------

            if global_step % 5 == 0:

                print("############################")
                print("epoch:", epoch)
                print("step:", global_step)

                print(
                    "loss:",
                    loss.item()
                )

                print(
                    "batch accuracy:",
                    accuracy.item()
                )

                print(
                    "gradient norm:",
                    total_grad_norm
                )

            if global_step % 100 == 0:
                writer.flush()

            global_step += 1

        # =====================================================
        # TRAINING EPOCH STATISTICS
        # =====================================================

        train_epoch_loss = (
            train_loss_sum / train_samples
        )

        train_epoch_accuracy = (
            train_correct / train_samples
        )

        # =====================================================
        # VALIDATION
        # =====================================================

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            normalize_255=False
        )

        # -----------------------------------------
        # TensorBoard epoch metrics
        # -----------------------------------------

        writer.add_scalar(
            "epoch/train_loss",
            train_epoch_loss,
            epoch
        )

        writer.add_scalar(
            "epoch/train_accuracy",
            train_epoch_accuracy,
            epoch
        )

        writer.add_scalar(
            "validation/loss",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "validation/accuracy",
            val_accuracy,
            epoch
        )

        print("")
        print("================================")
        print(f"Epoch {epoch}")
        print("--------------------------------")
        print(
            f"Train Loss:     "
            f"{train_epoch_loss:.4f}"
        )
        print(
            f"Train Accuracy: "
            f"{train_epoch_accuracy:.4f}"
        )
        print(
            f"Val Loss:       "
            f"{val_loss:.4f}"
        )
        print(
            f"Val Accuracy:   "
            f"{val_accuracy:.4f}"
        )
        print("================================")

        # =====================================================
        # SAVE BEST MODEL
        # =====================================================

        improved = False

        # Primary metric = validation accuracy
        if val_accuracy > best_val_accuracy:

            improved = True

        # If accuracy is exactly equal,
        # prefer smaller validation loss
        elif (
            val_accuracy == best_val_accuracy
            and val_loss < best_val_loss
        ):

            improved = True

        if improved:

            best_val_accuracy = val_accuracy
            best_val_loss = val_loss

            checkpoint = {

                "epoch": epoch,

                "global_step": global_step,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "val_loss":
                    val_loss,

                "train_accuracy":
                    train_epoch_accuracy,

                "train_loss":
                    train_epoch_loss
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print("")
            print("***** BEST MODEL UPDATED *****")
            print(
                f"Validation Accuracy: "
                f"{val_accuracy:.4f}"
            )
            print(
                f"Validation Loss: "
                f"{val_loss:.4f}"
            )
            print(
                f"Saved to: "
                f"{checkpoint_path}"
            )
            print("******************************")
            print("")

        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
