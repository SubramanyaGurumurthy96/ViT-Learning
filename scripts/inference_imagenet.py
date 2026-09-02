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
from vit.data import load_imagenette


# -----------------------------------------
# Model / data configuration.
# These MUST match the values used in
# scripts/train_imagenette.py.
# -----------------------------------------

IMAGE_SIZE = 160
PATCH_SIZE = 16
EMBEDDING_DIM = 128
NUM_HEADS = 4
MLP_DIM = 512
NUM_LAYERS = 4
NUM_CLASSES = 10

NUM_IMAGES_TO_SHOW = 9

# -----------------------------------------
# Human-readable names for the Imagenette
# WordNet-ID folders. ImageFolder assigns
# class indices by sorted folder name, so
# this list is ordered to match indices 0-9.
# -----------------------------------------

WORDNET_TO_LABEL = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def find_data_dir():
    candidates = [
        os.path.join(PROJECT_ROOT, "imageNette", "imagenette2-160"),
        os.path.join(PROJECT_ROOT, "imageNette"),
        os.path.join(PROJECT_ROOT, "imagenette2-160"),
    ]

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "val")):
            return candidate

    raise FileNotFoundError(
        "Could not locate an Imagenette dataset with a 'val' folder. "
        f"Looked in: {candidates}"
    )


def find_checkpoint():
    candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints", "best_vit_imagenette.pth"),
        os.path.join(SCRIPT_DIR, "checkpoints", "best_vit_imagenette.pth"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find a trained Imagenette checkpoint. "
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
    )

    return model


def class_names_for(dataset):
    # dataset.classes is the sorted list of WordNet-ID
    # folder names, indexed the same way as the labels.
    return [
        WORDNET_TO_LABEL.get(wordnet_id, wordnet_id)
        for wordnet_id in dataset.classes
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
            fontsize=10,
        )

    for j in range(NUM_IMAGES_TO_SHOW, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "ViT Imagenette Inference (green = correct, red = wrong)",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(PROJECT_ROOT, "inference_results.png")
    fig.savefig(output_path, dpi=120)

    print(f"Saved visualization to: {output_path}")

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

    _, val_dataset = load_imagenette(data_dir, image_size=IMAGE_SIZE)
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

    images, labels = next(iter(loader))

    with torch.no_grad():
        logits = model(images.float().to(device))
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

    predictions = predictions.cpu()
    confidences = confidences.cpu()

    print("")
    print("================ Predictions ================")
    for i in range(images.size(0)):
        true_name = names[labels[i].item()]
        pred_name = names[predictions[i].item()]
        mark = "OK " if labels[i] == predictions[i] else "XX "

        print(
            f"{mark} pred: {pred_name:<18} "
            f"({confidences[i].item() * 100:5.1f}%)  "
            f"true: {true_name}"
        )
    print("=============================================")
    print("")

    visualize(
        images,
        labels.tolist(),
        predictions.tolist(),
        confidences.tolist(),
        names,
    )


if __name__ == "__main__":
    main()
