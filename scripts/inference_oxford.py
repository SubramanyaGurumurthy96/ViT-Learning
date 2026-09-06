import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# -----------------------------------------
# Make the local `vit` package importable
# regardless of where this script is run from.
# -----------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vit import ViT
from vit.data import load_oxford_pets


# -----------------------------------------
# Model / data configuration.
# These MUST match the values used in
# scripts/train_oxford.py.
# -----------------------------------------

IMAGE_SIZE = 160
PATCH_SIZE = 16
EMBEDDING_DIM = 128
NUM_HEADS = 4
MLP_DIM = 512
NUM_LAYERS = 4
NUM_CLASSES = 37
NUM_SEGMENTS = 3

# Number of samples shown in the classification grid.
NUM_IMAGES_TO_SHOW = 9

# Number of samples shown in the segmentation panel.
NUM_SEG_TO_SHOW = 4

# Predictions with a confidence below this value are
# additionally shown in a separate visualization.
CONFIDENCE_THRESHOLD = 0.80

# Human-readable names for the segmentation trimap classes.
# The dataset loader remaps the raw trimap {1, 2, 3} to {0, 1, 2}.
SEGMENT_NAMES = ["foreground", "background", "boundary"]


def find_data_dir():
    candidates = [
        os.path.join(PROJECT_ROOT, "oxford_data"),
    ]

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "images")) and \
                os.path.isdir(os.path.join(candidate, "annotations")):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Oxford-IIIT Pet dataset "
        "(expected 'images/' and 'annotations/' folders). "
        f"Looked in: {candidates}"
    )


def find_checkpoint():
    candidates = [
        os.path.join(SCRIPT_DIR, "checkpoints", "best_vit_oxford.pth"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find a trained Oxford checkpoint. "
        f"Looked in: {candidates}"
    )


def build_model():
    model = ViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        num_segments=NUM_SEGMENTS,
    )

    return model


def class_names_for(dataset):
    # Each entry is (name, label) where name looks like
    # "Abyssinian_100" or "great_pyrenees_23". The breed is
    # everything before the trailing "_<number>".
    label_to_breed = {}

    for name, label in dataset.entries:
        if label not in label_to_breed:
            breed = name.rsplit("_", 1)[0]
            label_to_breed[label] = breed.replace("_", " ")

    return [
        label_to_breed.get(i, f"class_{i}")
        for i in range(NUM_CLASSES)
    ]


def visualize(images, true_labels, pred_labels, confidences, names):
    grid = int(NUM_IMAGES_TO_SHOW ** 0.5)

    fig, axes = plt.subplots(grid, grid, figsize=(3 * grid, 3 * grid))
    axes = axes.flatten()

    for i in range(NUM_IMAGES_TO_SHOW):
        ax = axes[i]

        # image tensor is [C, H, W] in [0, 1]
        image = images[i].permute(1, 2, 0).cpu().numpy()

        ax.imshow(image)
        ax.axis("off")

        true_name = names[true_labels[i]]
        pred_name = names[pred_labels[i]]
        correct = true_labels[i] == pred_labels[i]

        color = "green" if correct else "red"

        ax.set_title(
            f"pred: {pred_name} ({confidences[i] * 100:.1f}%)\n"
            f"true: {true_name}",
            color=color,
            fontsize=9,
        )

    for j in range(NUM_IMAGES_TO_SHOW, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "ViT Oxford-Pets Classification "
        "(green = correct, red = wrong)",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(PROJECT_ROOT, "inference_oxford_results.png")
    fig.savefig(output_path, dpi=120)

    print(f"Saved classification visualization to: {output_path}")

    plt.show()


def visualize_segmentation(
    images, true_masks, pred_masks, true_labels, pred_labels, names
):
    count = min(NUM_SEG_TO_SHOW, images.size(0))

    fig, axes = plt.subplots(
        count,
        3,
        figsize=(3 * 3, 3 * count),
        squeeze=False,
    )

    for i in range(count):
        # image tensor is [C, H, W] in [0, 1]
        image = images[i].permute(1, 2, 0).cpu().numpy()

        gt_mask = true_masks[i].cpu().numpy()
        pred_mask = pred_masks[i].cpu().numpy()

        axes[i][0].imshow(image)
        axes[i][0].axis("off")

        true_name = names[true_labels[i]]
        pred_name = names[pred_labels[i]]
        correct = true_labels[i] == pred_labels[i]
        color = "green" if correct else "red"

        axes[i][0].set_title(
            f"pred: {pred_name}\ntrue: {true_name}",
            color=color,
            fontsize=9,
        )

        axes[i][1].imshow(
            gt_mask, cmap="viridis", vmin=0, vmax=NUM_SEGMENTS - 1
        )
        axes[i][1].axis("off")
        axes[i][1].set_title("ground-truth mask", fontsize=9)

        axes[i][2].imshow(
            pred_mask, cmap="viridis", vmin=0, vmax=NUM_SEGMENTS - 1
        )
        axes[i][2].axis("off")
        axes[i][2].set_title("predicted mask", fontsize=9)

    fig.suptitle(
        "ViT Oxford-Pets Segmentation "
        f"({', '.join(SEGMENT_NAMES)})",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(
        PROJECT_ROOT, "inference_oxford_segmentation.png"
    )
    fig.savefig(output_path, dpi=120)

    print(f"Saved segmentation visualization to: {output_path}")

    plt.show()


def visualize_low_confidence(
    images, true_labels, pred_labels, confidences, names, threshold
):
    low_idx = [
        i for i, conf in enumerate(confidences) if conf < threshold
    ]

    if not low_idx:
        print(
            f"No predictions below "
            f"{threshold * 100:.0f}% confidence."
        )
        return

    count = len(low_idx)
    cols = min(3, count)
    rows = (count + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for position, idx in enumerate(low_idx):
        ax = axes[position]

        # image tensor is [C, H, W] in [0, 1]
        image = images[idx].permute(1, 2, 0).cpu().numpy()

        ax.imshow(image)
        ax.axis("off")

        true_name = names[true_labels[idx]]
        pred_name = names[pred_labels[idx]]
        correct = true_labels[idx] == pred_labels[idx]

        color = "green" if correct else "red"

        ax.set_title(
            f"pred: {pred_name} ({confidences[idx] * 100:.1f}%)\n"
            f"true: {true_name}",
            color=color,
            fontsize=9,
        )

    for j in range(count, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Low-confidence predictions (< {threshold * 100:.0f}%)",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(
        PROJECT_ROOT, "inference_oxford_low_confidence.png"
    )
    fig.savefig(output_path, dpi=120)

    print(f"Saved low-confidence visualization to: {output_path}")

    plt.show()


def visualize_high_confidence(
    images, true_labels, pred_labels, confidences, names, threshold
):
    high_idx = [
        i for i, conf in enumerate(confidences) if conf >= threshold
    ]

    if not high_idx:
        print(
            f"No predictions at or above "
            f"{threshold * 100:.0f}% confidence."
        )
        return

    count = len(high_idx)
    cols = min(3, count)
    rows = (count + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        squeeze=False,
    )
    axes = axes.flatten()

    for position, idx in enumerate(high_idx):
        ax = axes[position]

        # image tensor is [C, H, W] in [0, 1]
        image = images[idx].permute(1, 2, 0).cpu().numpy()

        ax.imshow(image)
        ax.axis("off")

        true_name = names[true_labels[idx]]
        pred_name = names[pred_labels[idx]]
        correct = true_labels[idx] == pred_labels[idx]

        color = "green" if correct else "red"

        ax.set_title(
            f"pred: {pred_name} ({confidences[idx] * 100:.1f}%)\n"
            f"true: {true_name}",
            color=color,
            fontsize=9,
        )

    for j in range(count, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"High-confidence predictions (>= {threshold * 100:.0f}%)",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(
        PROJECT_ROOT, "inference_oxford_high_confidence.png"
    )
    fig.savefig(output_path, dpi=120)

    print(f"Saved high-confidence visualization to: {output_path}")

    plt.show()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------
    # Data
    # -----------------------------------------

    data_dir = find_data_dir()
    print(f"Loading data from: {data_dir}")

    _, val_dataset = load_oxford_pets(data_dir, image_size=IMAGE_SIZE)
    names = class_names_for(val_dataset)

    loader = DataLoader(
        val_dataset,
        batch_size=NUM_IMAGES_TO_SHOW,
        shuffle=True,
    )

    # -----------------------------------------
    # Model + weights
    # -----------------------------------------

    checkpoint_path = find_checkpoint()
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if "val_accuracy" in checkpoint:
        print(
            f"Checkpoint val accuracy: "
            f"{checkpoint['val_accuracy']:.4f}"
        )

    # -----------------------------------------
    # Inference on one random batch
    # -----------------------------------------

    images, labels, masks = next(iter(loader))

    with torch.no_grad():
        class_logits, seg_logits = model(images.float().to(device))
        probs = torch.softmax(class_logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        seg_predictions = seg_logits.argmax(dim=1)

    predictions = predictions.cpu()
    confidences = confidences.cpu()
    seg_predictions = seg_predictions.cpu()

    # Pixel-level segmentation accuracy for this batch
    seg_accuracy = (
        (seg_predictions == masks).float().mean().item()
    )

    print("")
    print("================ Predictions ================")
    for i in range(images.size(0)):
        true_name = names[labels[i].item()]
        pred_name = names[predictions[i].item()]
        mark = "OK " if labels[i] == predictions[i] else "XX "

        print(
            f"{mark} pred: {pred_name:<22} "
            f"({confidences[i].item() * 100:5.1f}%)  "
            f"true: {true_name}"
        )
    print("---------------------------------------------")
    print(f"batch segmentation pixel accuracy: {seg_accuracy:.4f}")
    print("=============================================")
    print("")

    # -----------------------------------------
    # Classification visualization
    # -----------------------------------------

    visualize(
        images,
        labels.tolist(),
        predictions.tolist(),
        confidences.tolist(),
        names,
    )

    # -----------------------------------------
    # Segmentation visualization (image vs
    # ground-truth mask vs predicted mask)
    # -----------------------------------------

    visualize_segmentation(
        images,
        masks,
        seg_predictions,
        labels.tolist(),
        predictions.tolist(),
        names,
    )

    # -----------------------------------------
    # Extra: predictions the model was unsure
    # about (confidence < CONFIDENCE_THRESHOLD)
    # get their own separate visualization.
    # -----------------------------------------

    confidence_list = confidences.tolist()

    low_conf_count = sum(
        1 for conf in confidence_list if conf < CONFIDENCE_THRESHOLD
    )

    print(
        f"{low_conf_count} of {len(confidence_list)} predictions "
        f"below {CONFIDENCE_THRESHOLD * 100:.0f}% confidence."
    )

    visualize_low_confidence(
        images,
        labels.tolist(),
        predictions.tolist(),
        confidence_list,
        names,
        CONFIDENCE_THRESHOLD,
    )

    # -----------------------------------------
    # Extra: the reverse case -- predictions the
    # model was confident about (confidence >=
    # CONFIDENCE_THRESHOLD) get their own
    # separate visualization too.
    # -----------------------------------------

    high_conf_count = sum(
        1 for conf in confidence_list if conf >= CONFIDENCE_THRESHOLD
    )

    print(
        f"{high_conf_count} of {len(confidence_list)} predictions "
        f"at or above {CONFIDENCE_THRESHOLD * 100:.0f}% confidence."
    )

    visualize_high_confidence(
        images,
        labels.tolist(),
        predictions.tolist(),
        confidence_list,
        names,
        CONFIDENCE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
