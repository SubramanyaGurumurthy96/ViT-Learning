import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from vit import ViT
from vit.config import MODEL_SHAPE_DEBUG
from vit.data import load_cifar10
from vit.utils import get_gradient_norm, validate


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT on CIFAR-10")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(
            PROJECT_ROOT, "cifar-10-python", "cifar-10-batches-py"
        ),
        help="Directory containing CIFAR-10 data_batch_* files",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint-path",
        default=os.path.join(SCRIPT_DIR, "checkpoints", "best_vit.pth"),
    )
    parser.add_argument(
        "--runs-dir",
        default=os.path.join(PROJECT_ROOT, "runs", "vit_experiment"),
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

    labels, images = load_cifar10(args.data_dir)

    # ----------------------------
    # NumPy -> PyTorch
    # ----------------------------

    images = torch.tensor(
        images,
        dtype=torch.float32
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    if MODEL_SHAPE_DEBUG:
        print("before permute:", images.shape)

    # [B, H, W, C] -> [B, C, H, W]
    images = images.permute(0, 3, 1, 2)

    # CIFAR pixels are 0..255
    images = images / 255.0

    if MODEL_SHAPE_DEBUG:
        print("images:", images.shape)
        print("labels:", labels.shape)

    # ----------------------------
    # Create Dataset
    # ----------------------------

    dataset = TensorDataset(
        images,
        labels
    )

    train_size = 45000
    val_size = 5000

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ----------------------------
    # Create DataLoader
    # ----------------------------

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = ViT(
        image_size=32,
        patch_size=4,
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

            batch_images = batch_images.float() / 255.0

            # -----------------------------------------
            # Clear previous gradients
            # -----------------------------------------

            optimizer.zero_grad(set_to_none=True)

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
            # Attention visualization
            # -----------------------------------------

            if global_step % 100 == 0:

                for block_idx, block in enumerate(model.blocks):

                    attention = (
                        block.attention.last_attention
                    )

                    # First image
                    # Head 0
                    # CLS -> 64 image patches

                    cls_attention = attention[
                        0,
                        0,
                        0,
                        1:
                    ]

                    heatmap = cls_attention.reshape(
                        8,
                        8
                    )

                    fig, ax = plt.subplots(
                        figsize=(4, 4)
                    )

                    im = ax.imshow(
                        heatmap
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    ax.set_title(
                        f"Epoch {epoch} "
                        f"- Block {block_idx} "
                        f"- Head 0"
                    )

                    fig.colorbar(im)

                    writer.add_figure(
                        f"attention/"
                        f"block_{block_idx}_head_0",
                        fig,
                        global_step
                    )

                    plt.close(fig)

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

            optimizer.step()

            # -----------------------------------------
            # Terminal output
            # -----------------------------------------

            if global_step % 100 == 0:

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
            criterion
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
