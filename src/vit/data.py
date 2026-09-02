import os
import pickle

import numpy as np
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
