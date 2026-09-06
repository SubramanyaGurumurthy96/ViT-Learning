import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def load_cifar10(data_dir):
    all_images = []
    all_labels = []

    for i in range(1, 6):

        file_path = os.path.join(data_dir, f"data_batch_{i}")

        with open(file_path, "rb") as fo:
            data_dict = pickle.load(fo, encoding="bytes")

        raw_data = data_dict[b"data"]
        labels = data_dict[b"labels"]

        images = raw_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

        all_images.append(images)
        all_labels.extend(labels)

    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.array(all_labels)

    print("images:", all_images.shape)
    print("labels:", all_labels.shape)

    return all_labels, all_images


def load_imagenette(data_dir, image_size=160):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), eval_tf)
    return train_ds, val_ds


class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet dataset with classification + segmentation.

    Each item is a tuple ``(image, label, mask)`` where:

    - ``image``: float tensor ``[3, image_size, image_size]`` in ``[0, 1]``.
    - ``label``: int breed class in ``[0, 36]`` (the original 1-37
      CLASS-ID from ``list.txt``, shifted to be 0-indexed).
    - ``mask``:  long tensor ``[image_size, image_size]`` segmentation
      trimap. With ``remap_mask=True`` the original trimap values
      ``{1: foreground, 2: background, 3: boundary}`` are shifted to
      ``{0: foreground, 1: background, 2: boundary}``.

    Only samples that have BOTH an input image and a segmentation
    trimap are included.
    """

    def __init__(self, data_dir, entries, image_size=160, remap_mask=True):
        self.images_dir = os.path.join(data_dir, "images")
        self.trimaps_dir = os.path.join(
            data_dir, "annotations", "trimaps"
        )
        self.entries = entries
        self.image_size = image_size
        self.remap_mask = remap_mask

        self.image_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        name, label = self.entries[idx]

        image_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.trimaps_dir, name + ".png")

        # ----- input image -----
        image = Image.open(image_path).convert("RGB")
        image = self.image_tf(image)

        # ----- segmentation trimap -----
        # Nearest-neighbour resize so the integer class
        # values in the trimap are preserved.
        mask = Image.open(mask_path).resize(
            (self.image_size, self.image_size),
            Image.NEAREST,
        )
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        if self.remap_mask:
            # {1, 2, 3} -> {0, 1, 2}
            mask = mask - 1

        return image, label, mask


def load_oxford_pets(data_dir, image_size=160, remap_mask=True):
    """Load the Oxford-IIIT Pet dataset.

    Returns ``(train_ds, val_ds)`` where each dataset yields
    ``(image, label, mask)`` samples that carry both the
    classification label and the segmentation trimap.

    ``data_dir`` should point at the folder containing ``images/`` and
    ``annotations/`` (e.g. ``.../oxford_data``). The train/val split
    follows the dataset's ``trainval.txt`` / ``test.txt`` files.
    """

    ann_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "images")
    trimaps_dir = os.path.join(ann_dir, "trimaps")

    def read_split(split_file):
        entries = []

        split_path = os.path.join(ann_dir, split_file)

        with open(split_path) as f:
            for line in f:
                line = line.strip()

                # skip blank lines and the '#' header comments
                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                name = parts[0]
                class_id = int(parts[1])

                # require BOTH the input image and its
                # segmentation trimap to be present
                image_path = os.path.join(images_dir, name + ".jpg")
                mask_path = os.path.join(trimaps_dir, name + ".png")

                if not os.path.exists(image_path):
                    continue
                if not os.path.exists(mask_path):
                    continue

                # store label 0-indexed (original CLASS-ID is 1-37)
                entries.append((name, class_id - 1))

        return entries

    train_entries = read_split("trainval.txt")
    val_entries = read_split("test.txt")

    train_ds = OxfordPetDataset(
        data_dir, train_entries, image_size, remap_mask
    )
    val_ds = OxfordPetDataset(
        data_dir, val_entries, image_size, remap_mask
    )

    return train_ds, val_ds
