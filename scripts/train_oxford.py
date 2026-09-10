import argparse
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vit import ViT
from vit.data import load_oxford_pets
from vit.utils import (
    find_nonfinite_gradients,
    get_gradient_norm,
    optimizer_state_is_finite,
    resolve_device,
    state_dict_is_finite,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def dice_loss(logits, targets, num_classes=3, eps=1e-6):

    # logits:
    # [B, 3, H, W]

    probs = torch.softmax(logits, dim=1)

    # targets:
    # [B, H, W]

    targets_one_hot = F.one_hot(
        targets,
        num_classes=num_classes
    )

    # [B, H, W, 3]
    # -> [B, 3, H, W]

    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    intersection = (
        probs * targets_one_hot
    ).sum(dim=(0, 2, 3))

    denominator = (
        probs + targets_one_hot
    ).sum(dim=(0, 2, 3))

    dice = (
        2 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1 - dice.mean()


def mixup_batch(images, alpha):
    # -----------------------------------------
    # Mixup (Zhang et al., 2018).
    #
    # The batch is blended with a shuffled copy of
    # itself under a single lambda drawn from
    # Beta(alpha, alpha) - one lambda per batch, as
    # in the paper, not one per sample.
    #
    # Only the INPUT is mixed here. The two target
    # sets are combined in the loss (see mix_loss),
    # which is what makes mixup a regularizer
    # rather than a relabelling scheme.
    # -----------------------------------------

    lam = float(
        torch.distributions.Beta(alpha, alpha).sample()
    )

    index = torch.randperm(
        images.size(0),
        device=images.device
    )

    mixed_images = (
        lam * images + (1.0 - lam) * images[index]
    )

    return mixed_images, index, lam


def mix_loss(loss_fn, outputs, target_a, target_b, lam):
    # -----------------------------------------
    # lambda-weighted loss against both target sets.
    #
    # Works for any loss taking (outputs, hard
    # targets): the classification CE, the
    # segmentation CE, and the soft Dice above.
    #
    # At lam == 1.0 (mixup off for this batch) it
    # collapses to the original single-target loss,
    # so a disabled-mixup run is bit-for-bit the
    # old objective and costs nothing extra.
    # -----------------------------------------

    if lam >= 1.0:
        return loss_fn(outputs, target_a)

    return (
        lam * loss_fn(outputs, target_a)
        + (1.0 - lam) * loss_fn(outputs, target_b)
    )


def cosine_lr_at(base_lr, eta_min, epoch, t_max):
    # -----------------------------------------
    # Closed form of CosineAnnealingLR, used to
    # re-synchronise the optimizer's LR after a
    # resume. Stepping the scheduler forward in a
    # loop would also work, but every step writes
    # into optimizer.param_groups, so the closed
    # form is both cheaper and easier to verify.
    # -----------------------------------------

    return eta_min + (base_lr - eta_min) * (
        1.0 + math.cos(math.pi * epoch / t_max)
    ) / 2.0


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


def validate(
    model,
    val_loader,
    classification_criterion,
    segmentation_criterion,
    num_segments=3,
    device=None,
):
    # -----------------------------------------
    # Validation loop for the joint
    # classification + segmentation model.
    #
    # The model returns a tuple
    # (class_logits, seg_logits), so we can't
    # reuse vit.utils.validate here.
    #
    # The loss here MUST match the training
    # objective: classification CE + (seg CE + Dice).
    # -----------------------------------------

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    correct_pixels = 0
    total_pixels = 0

    # Per-class accumulators for mean IoU and per-class Dice.
    # Class ids (remapped trimap):
    #   0 = foreground, 1 = background, 2 = boundary
    intersection = torch.zeros(num_segments)
    union = torch.zeros(num_segments)
    pred_area = torch.zeros(num_segments)
    target_area = torch.zeros(num_segments)

    with torch.no_grad():

        for batch_images, batch_labels, batch_masks in val_loader:

            batch_images = batch_images.float()

            if device is not None:
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_masks = batch_masks.to(device, non_blocking=True)

            class_logits, seg_logits = model(batch_images)

            classification_loss = classification_criterion(
                class_logits,
                batch_labels
            )

            # Match the TRAINING objective exactly:
            # segmentation = cross-entropy + Dice.
            seg_ce_loss = segmentation_criterion(
                seg_logits,
                batch_masks
            )

            seg_dice_loss = dice_loss(
                seg_logits,
                batch_masks,
                num_classes=num_segments
            )

            segmentation_loss = seg_ce_loss + seg_dice_loss

            loss = classification_loss + segmentation_loss

            predictions = class_logits.argmax(dim=1)

            total_loss += loss.item() * batch_images.size(0)

            total_correct += (
                predictions == batch_labels
            ).sum().item()

            total_samples += batch_images.size(0)

            # pixel-level segmentation accuracy
            seg_predictions = seg_logits.argmax(dim=1)

            correct_pixels += (
                seg_predictions == batch_masks
            ).sum().item()

            total_pixels += batch_masks.numel()

            # per-class IoU / Dice accumulation
            for c in range(num_segments):
                pred_c = (seg_predictions == c)
                target_c = (batch_masks == c)

                intersection[c] += (pred_c & target_c).sum().item()
                union[c] += (pred_c | target_c).sum().item()
                pred_area[c] += pred_c.sum().item()
                target_area[c] += target_c.sum().item()

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    seg_accuracy = correct_pixels / total_pixels

    eps = 1e-6

    per_class_iou = (
        intersection / (union + eps)
    ).tolist()

    mean_iou = sum(per_class_iou) / num_segments

    per_class_dice = (
        (2.0 * intersection) / (pred_area + target_area + eps)
    ).tolist()

    return (
        avg_loss,
        accuracy,
        seg_accuracy,
        mean_iou,
        per_class_iou,
        per_class_dice,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ViT on Oxford-IIIT Pets "
        "(classification + segmentation)"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "oxford_data"),
        help="Directory containing images/ and annotations/ folders",
    )
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-path",
        default=os.path.join(
            SCRIPT_DIR, "checkpoints", "best_vit_oxford.pth"
        ),
    )
    parser.add_argument(
        "--runs-dir",
        default=os.path.join(PROJECT_ROOT, "runs", "vit_oxford"),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' (CUDA when available), 'cuda', 'cuda:1', 'cpu', ...",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load --checkpoint-path before training. OFF by default: "
        "a checkpoint saved from a diverged run carries NaN weights "
        "and NaN AdamW moments, and resuming from it poisons the new "
        "run at step 0. Only a verifiably finite checkpoint is "
        "accepted.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.4,
        help="Beta(alpha, alpha) parameter for Mixup. 0 disables it "
        "and restores the previous objective exactly. Larger = more "
        "mixing = stronger regularization; 0.2 is a gentle start, "
        "0.4-0.8 is the next dial to turn.",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=0.4,
        help="Fraction of training batches that get mixed. Lower it "
        "(e.g. 0.5) to leave a share of batches un-mixed, which also "
        "gives the segmentation head clean supervision on those steps.",
    )
    parser.add_argument(
        "--grad-norm-warn",
        type=float,
        default=1e4,
        help="Print a warning when the global gradient norm exceeds "
        "this. Diagnostic only - nothing is clipped or rescaled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(
        os.path.dirname(args.checkpoint_path) or ".",
        exist_ok=True
    )

    checkpoint_path = args.checkpoint_path

    # -----------------------------------------
    # Device
    #
    # Everything - model and every batch - has
    # to be moved explicitly, otherwise the run
    # silently stays on the CPU.
    # -----------------------------------------

    device = resolve_device(args.device)

    print(f"device: {device}")

    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
        print(f"cuda devices visible: {torch.cuda.device_count()}")
    elif args.device == "auto":
        print("cuda not available - running on CPU")

    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    train_dataset, val_dataset = load_oxford_pets(
        args.data_dir,
        image_size=args.image_size
    )

    # -----------------------------------------
    # Stronger training-time augmentation.
    #
    # Crop + horizontal flip are applied with the
    # SAME random parameters to both the image and
    # its segmentation mask (handled inside the
    # dataset), so they stay spatially aligned.
    # Color jitter is photometric and image-only.
    # -----------------------------------------

    train_dataset.augment = True

    pin_memory = device.type == "cuda"

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    # -----------------------------------------
    # Oxford-IIIT Pets has 37 breed classes and
    # the trimaps have 3 segmentation classes.
    # -----------------------------------------

    model = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embedding_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
        num_classes=37,
        num_segments=3
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # -----------------------------------------
    # Resume - opt-in, and validated.
    #
    # This used to load unconditionally whenever
    # the file existed. A diverged run had
    # already written a checkpoint whose weights
    # AND AdamW moments were NaN, so every later
    # run silently started from NaN and could
    # never recover. Restoring the weights alone
    # would not be enough either: a NaN exp_avg_sq
    # re-poisons the first step.
    #
    # A resume also has to restore the *schedule*,
    # not just the tensors. Previously it did not,
    # with three consequences visible in the last
    # run's log: epoch 90 of a cosine schedule
    # restarted at the full base LR, the epoch
    # counter restarted at 0, and best_val_accuracy
    # restarted at 0.0 - so the very first epoch
    # after the resume (0.5072) overwrote a better
    # checkpoint (0.5129).
    # -----------------------------------------

    start_epoch = 0
    global_step = 0

    if not args.resume:
        if os.path.exists(checkpoint_path):
            print(
                f"not resuming (pass --resume to load "
                f"{checkpoint_path})"
            )
    elif not os.path.exists(checkpoint_path):
        print(
            f"--resume given but no checkpoint at {checkpoint_path} "
            f"- starting from scratch"
        )
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model_ok, bad_param = state_dict_is_finite(
            checkpoint["model_state_dict"]
        )

        optim_ok, bad_moment = optimizer_state_is_finite(
            checkpoint["optimizer_state_dict"]
        )

        if not model_ok or not optim_ok:
            raise RuntimeError(
                f"Refusing to resume from {checkpoint_path}: it "
                f"contains non-finite values "
                f"(model: {bad_param}, optimizer: {bad_moment}). "
                f"This checkpoint came from a diverged run - delete "
                f"it or train without --resume."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))

        # The checkpoint is only ever written on an improvement, so
        # its metrics ARE the best so far. Carrying them over is what
        # stops the first resumed epoch from clobbering a better model.

        best_val_accuracy = float(
            checkpoint.get("val_accuracy", 0.0)
        )

        best_val_loss = float(
            checkpoint.get("val_loss", float("inf"))
        )

        # -----------------------------------------
        # Scheduler
        # -----------------------------------------

        scheduler_state = checkpoint.get("scheduler_state_dict")

        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        else:
            print(
                "checkpoint carries no scheduler state (written before "
                "scheduler resume existed) - rebuilding the cosine "
                "position from the saved epoch instead"
            )

        # --epochs and --lr describe THIS run, and the loop below runs
        # range(start_epoch, args.epochs), so the cosine horizon has to
        # agree with them. load_state_dict() restored the SAVED T_max
        # and base_lrs, which would otherwise silently outrank the flags.

        if scheduler.T_max != args.epochs:
            print(
                f"cosine T_max {scheduler.T_max} -> {args.epochs} "
                f"(from --epochs)"
            )

            scheduler.T_max = args.epochs

        if any(base_lr != args.lr for base_lr in scheduler.base_lrs):
            print(
                f"cosine base LR {scheduler.base_lrs} -> {args.lr} "
                f"(from --lr)"
            )

            scheduler.base_lrs = [
                args.lr for _ in optimizer.param_groups
            ]

        # Same story for weight decay, which came back inside the
        # optimizer's param_groups.

        for group in optimizer.param_groups:
            if group["weight_decay"] != args.weight_decay:
                print(
                    f"weight decay {group['weight_decay']} -> "
                    f"{args.weight_decay} (from --weight-decay)"
                )

            group["weight_decay"] = args.weight_decay

        # The saved position is "partway through epoch <saved epoch>",
        # because the checkpoint is written before scheduler.step().
        # Training resumes one epoch later, so move the curve to where
        # that step() would have left it and push the LR into the
        # optimizer now - otherwise the resumed epoch trains at the
        # previous epoch's learning rate.

        scheduler.last_epoch = start_epoch
        scheduler._step_count = start_epoch + 1

        resumed_lrs = [
            cosine_lr_at(
                base_lr,
                scheduler.eta_min,
                scheduler.last_epoch,
                scheduler.T_max
            )
            for base_lr in scheduler.base_lrs
        ]

        for group, group_lr in zip(optimizer.param_groups, resumed_lrs):
            group["lr"] = group_lr

        scheduler._last_lr = resumed_lrs

        print(
            f"Checkpoint loaded (epoch {checkpoint.get('epoch')}, "
            f"val_accuracy {checkpoint.get('val_accuracy')})"
        )

        print(
            f"resuming at epoch {start_epoch}/{args.epochs}, "
            f"global_step {global_step}, lr {resumed_lrs[0]:.6g}"
        )

        print(
            f"best so far: val_accuracy {best_val_accuracy:.4f}, "
            f"val_loss {best_val_loss:.4f}"
        )

    if start_epoch >= args.epochs:
        print(
            f"nothing to do: the checkpoint already finished epoch "
            f"{start_epoch - 1} and --epochs is {args.epochs}. Raise "
            f"--epochs to keep training."
        )

        return

    # -----------------------------------------
    # Two objectives: breed classification and
    # pixel-wise segmentation of the trimap.
    # -----------------------------------------

    classification_criterion = nn.CrossEntropyLoss()
    segmentation_criterion = nn.CrossEntropyLoss()

    print_once = False

    # -----------------------------------------
    # Mixup - the new regularizer for this run.
    #
    # Classification is what is overfitting
    # (train ~0.99 vs val ~0.51 for 90 epochs
    # straight), so that is what gets mixed.
    # -----------------------------------------

    mixup_enabled = args.mixup_alpha > 0.0

    if mixup_enabled:
        print(
            f"mixup: alpha={args.mixup_alpha}, prob={args.mixup_prob}"
        )
        print(
            "note: train accuracy below is mixup-weighted, so it "
            "reads much lower than an un-mixed run - expected, not a "
            "regression. Compare validation accuracy across runs."
        )
    else:
        print("mixup: disabled (--mixup-alpha 0)")

    writer = SummaryWriter(args.runs_dir)

    attention_grid = args.image_size // args.patch_size

    accumulatation_step = 1

    # Set when a non-finite loss or gradient is seen, so the run
    # stops instead of writing the corruption into the weights.
    stop_reason = None

    for epoch in range(start_epoch, args.epochs):

        # =====================================================
        # TRAINING
        # =====================================================

        model.train()

        train_loss_sum = 0.0
        train_correct = 0.0
        train_samples = 0

        for batch_idx, (
            batch_images,
            batch_labels,
            batch_masks,
        ) in enumerate(loader):

            # -----------------------------------------
            # Prepare input
            # -----------------------------------------

            batch_images = batch_images.float().to(
                device, non_blocking=pin_memory
            )

            batch_labels = batch_labels.to(
                device, non_blocking=pin_memory
            )

            batch_masks = batch_masks.to(
                device, non_blocking=pin_memory
            )

            # -----------------------------------------
            # Mixup
            #
            # The batch is blended with a shuffled
            # copy of itself, and BOTH label sets are
            # kept so the losses below can weight them
            # by lambda.
            #
            # The segmentation objective is unchanged
            # in form - still CE + Dice - but it sees
            # the same mixed pixels, so its two mask
            # sets get blended with the same lambda.
            # Supervising it with mask A alone would
            # ask the head to predict A from a picture
            # that is only partly A. Turn --mixup-prob
            # down to leave some batches un-mixed.
            # -----------------------------------------

            if (
                mixup_enabled
                and torch.rand(1).item() < args.mixup_prob
            ):
                model_input, mix_index, lam = mixup_batch(
                    batch_images,
                    args.mixup_alpha
                )

                labels_b = batch_labels[mix_index]
                masks_b = batch_masks[mix_index]
            else:
                model_input = batch_images
                lam = 1.0

                labels_b = batch_labels
                masks_b = batch_masks

            # -----------------------------------------
            # Forward pass
            # -----------------------------------------

            class_logits, seg_logits = model(model_input)

            predictions = class_logits.argmax(dim=1)

            # -----------------------------------------
            # Print shapes once
            # -----------------------------------------

            if not print_once:

                print("batch images:", batch_images.shape)
                print("batch labels:", batch_labels.shape)
                print("batch masks:", batch_masks.shape)
                print("class logits:", class_logits.shape)
                print("seg logits:", seg_logits.shape)

                print_once = True

            # -----------------------------------------
            # Loss (classification + segmentation)
            # -----------------------------------------

            classification_loss = mix_loss(
                classification_criterion,
                class_logits,
                batch_labels,
                labels_b,
                lam
            )

            seg_dice_loss = mix_loss(
                lambda outputs, targets: dice_loss(
                    outputs,
                    targets,
                    num_classes=3
                ),
                seg_logits,
                batch_masks,
                masks_b,
                lam
            )

            seg_ce_loss = mix_loss(
                segmentation_criterion,
                seg_logits,
                batch_masks,
                masks_b,
                lam
            )

            segmentation_loss = seg_ce_loss + seg_dice_loss
            loss = classification_loss + segmentation_loss
            # loss = classification_loss 

            loss = loss / accumulatation_step

            # -----------------------------------------
            # Fail fast on a non-finite loss.
            #
            # Checked BEFORE backward/step: once a nan
            # reaches the optimizer the weights are
            # unrecoverable, and every metric printed
            # afterwards is an artifact (argmax over
            # all-nan logits just returns index 0).
            # -----------------------------------------

            if not math.isfinite(loss.item()):
                stop_reason = (
                    f"non-finite loss at epoch {epoch}, "
                    f"step {global_step}: "
                    f"total={loss.item()} "
                    f"classification={classification_loss.item()} "
                    f"segmentation={segmentation_loss.item()} "
                    f"(seg_ce={seg_ce_loss.item()} "
                    f"seg_dice={seg_dice_loss.item()})"
                )
                break

            # -----------------------------------------
            # Accuracy
            #
            # Under mixup neither label set is "the"
            # target, so batch accuracy is the
            # lambda-weighted agreement with both -
            # the same convention as the loss. This is
            # deliberately NOT comparable to the train
            # accuracy of an un-mixed run.
            # -----------------------------------------

            correct_a = (
                predictions == batch_labels
            ).float().sum()

            if lam >= 1.0:
                batch_correct = correct_a
            else:
                batch_correct = (
                    lam * correct_a
                    + (1.0 - lam) * (
                        predictions == labels_b
                    ).float().sum()
                )

            accuracy = batch_correct / batch_images.size(0)

            # Accumulate epoch statistics

            train_loss_sum += (
                loss.item() * batch_images.size(0)
            )

            train_correct += batch_correct.item()

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
                "training/classification_loss",
                classification_loss.item(),
                global_step
            )

            writer.add_scalar(
                "training/segmentation_loss",
                segmentation_loss.item(),
                global_step
            )

            writer.add_scalar(
                "training/accuracy",
                accuracy.item(),
                global_step
            )

            writer.add_scalar(
                "training/mixup_lambda",
                lam,
                global_step
            )

            # -----------------------------------------
            # Layer heatmap visualization
            # -----------------------------------------

            if global_step % 5 == 0:

                # -----------------------------------------
                # Segmentation visualization
                #
                # Ground-truth trimap vs the model's
                # predicted segmentation for the first image.
                # -----------------------------------------

                log_heatmap(
                    writer,
                    "segmentation/ground_truth",
                    batch_masks[0],
                    f"Epoch {epoch} - GT Mask",
                    global_step
                )

                log_heatmap(
                    writer,
                    "segmentation/prediction",
                    seg_logits[0].argmax(dim=0),
                    f"Epoch {epoch} - Pred Mask",
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

            # -----------------------------------------
            # Fail fast on a non-finite gradient.
            #
            # get_gradient_norm accumulates in float64,
            # so - unlike an fp32 reduction, which
            # saturates at ~3.4e38 and reports inf for
            # merely enormous gradients - a non-finite
            # value here means real inf/nan elements.
            # find_nonfinite_gradients then names them.
            #
            # Nothing is clipped: the point is to stop
            # with the evidence intact, not to paper
            # over the divergence.
            # -----------------------------------------

            if not math.isfinite(total_grad_norm):
                offenders = find_nonfinite_gradients(
                    model.named_parameters()
                )

                detail = ", ".join(
                    f"{name} ({count}/{numel} non-finite)"
                    for name, count, numel in offenders[:10]
                ) or "none found - check the norm itself"

                stop_reason = (
                    f"non-finite gradient at epoch {epoch}, "
                    f"step {global_step}: "
                    f"float64 grad norm={total_grad_norm}; "
                    f"{len(offenders)} parameter tensor(s) affected: "
                    f"{detail}"
                )
                break

            if total_grad_norm > args.grad_norm_warn:
                print(
                    f"WARNING step {global_step}: gradient norm "
                    f"{total_grad_norm:.3e} exceeds "
                    f"{args.grad_norm_warn:.3e} - gradients are "
                    f"exploding even though the loss is still finite"
                )

            writer.add_scalar(
                "gradients/enc1",
                get_gradient_norm(model.enc1.parameters()),
                global_step
            )

            writer.add_scalar(
                "gradients/enc2",
                get_gradient_norm(model.enc2.parameters()),
                global_step
            )

            writer.add_scalar(
                "gradients/enc3",
                get_gradient_norm(model.enc3.parameters()),
                global_step
            )

            writer.add_scalar(
                "gradients/enc4",
                get_gradient_norm(model.enc4.parameters()),
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

            writer.add_scalar(
                "gradients/segmentation_head",
                get_gradient_norm(
                    model.segmentation_head.parameters()
                ),
                global_step
            )

            # -----------------------------------------
            # Update parameters
            # -----------------------------------------
            if (batch_idx + 1) % accumulatation_step == 0:
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
                    "classification loss:",
                    classification_loss.item()
                )

                print(
                    "segmentation loss:",
                    segmentation_loss.item()
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

        if stop_reason is not None:
            break

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

        (
            val_loss,
            val_accuracy,
            val_seg_accuracy,
            val_mean_iou,
            val_per_class_iou,
            val_per_class_dice,
        ) = validate(
            model,
            val_loader,
            classification_criterion,
            segmentation_criterion,
            device=device,
        )

        seg_class_names = ["foreground", "background", "boundary"]

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

        writer.add_scalar(
            "validation/segmentation_accuracy",
            val_seg_accuracy,
            epoch
        )

        writer.add_scalar(
            "validation/mean_iou",
            val_mean_iou,
            epoch
        )

        for c, cname in enumerate(seg_class_names):
            writer.add_scalar(
                f"validation/iou_{cname}",
                val_per_class_iou[c],
                epoch
            )
            writer.add_scalar(
                f"validation/dice_{cname}",
                val_per_class_dice[c],
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
        print(
            f"Val Seg Acc:    "
            f"{val_seg_accuracy:.4f}"
        )
        print(
            f"Val Mean IoU:   "
            f"{val_mean_iou:.4f}"
        )
        print(
            f"Per-class IoU:  "
            f"fg={val_per_class_iou[0]:.4f}  "
            f"bg={val_per_class_iou[1]:.4f}  "
            f"boundary={val_per_class_iou[2]:.4f}"
        )
        print(
            f"Per-class Dice: "
            f"fg={val_per_class_dice[0]:.4f}  "
            f"bg={val_per_class_dice[1]:.4f}  "
            f"boundary={val_per_class_dice[2]:.4f}"
        )
        print("================================")

        # =====================================================
        # SAVE BEST MODEL
        # =====================================================

        improved = False

        # -----------------------------------------
        # A diverged epoch must never be saved.
        #
        # best_val_accuracy starts at 0.0, so the
        # chance-level accuracy of a fully-NaN model
        # (1/37 = 0.027 > 0.0) counted as "improved"
        # and overwrote the checkpoint with NaN
        # weights - which the next run then loaded.
        # Accuracy alone cannot detect this; the
        # loss and the weights have to be checked.
        # -----------------------------------------

        metrics_finite = (
            math.isfinite(val_loss)
            and math.isfinite(val_accuracy)
            and math.isfinite(train_epoch_loss)
        )

        weights_finite, bad_param = state_dict_is_finite(
            model.state_dict()
        )

        if not metrics_finite or not weights_finite:
            print("")
            print("!!!!! CHECKPOINT SKIPPED !!!!!")
            print(
                f"Epoch {epoch} produced non-finite values "
                f"(val_loss={val_loss}, "
                f"val_accuracy={val_accuracy}, "
                f"first non-finite parameter={bad_param}). "
                f"Not saving - this model is unrecoverable and "
                f"saving it would poison the next run."
            )
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("")

            stop_reason = (
                f"non-finite validation at epoch {epoch}: "
                f"val_loss={val_loss}, "
                f"first non-finite parameter={bad_param}"
            )

            break

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

                # Saved BEFORE scheduler.step() for this epoch, so
                # last_epoch == epoch. The resume path accounts for
                # that and advances the curve by one.
                "scheduler_state_dict":
                    scheduler.state_dict(),

                "mixup_alpha":
                    args.mixup_alpha,

                "mixup_prob":
                    args.mixup_prob,

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

        writer.add_scalar(
            "epoch/learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch
        )

        writer.flush()

        # step the cosine LR schedule once per epoch
        scheduler.step()

    writer.flush()
    writer.close()

    if stop_reason is not None:
        print("")
        print("==================================================")
        print("TRAINING STOPPED - NON-FINITE VALUE DETECTED")
        print("--------------------------------------------------")
        print(stop_reason)
        print("--------------------------------------------------")
        print(
            "No checkpoint was written for this state. To find "
            "which tensor blows up first, run:"
        )
        print("    python3 diagnose_oxford.py")
        print("==================================================")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
