import torch


def get_gradient_norm(parameters, dtype=torch.float64):
    """Global L2 norm over all parameter gradients.

    The per-tensor reduction is accumulated in ``dtype``
    (float64 by default) rather than in the gradients' own
    fp32. This matters: an fp32 sum-of-squares saturates at
    ~3.4e38, so a *finite* gradient tensor with elements
    around 1e16-1e19 can make ``tensor.norm(2)`` return
    ``inf`` even though every element is finite. Accumulating
    in float64 removes that false positive, so a non-finite
    return value here means the gradients really are
    non-finite - see ``find_nonfinite_gradients``.

    (An fp32 gradient element cannot exceed 3.4e38, so even a
    few hundred parameter tensors at that magnitude sum to
    ~1e79 - comfortably inside float64 range.)
    """
    total = 0.0

    for p in parameters:
        if p.grad is not None:
            grad_norm = torch.linalg.vector_norm(
                p.grad.detach(),
                ord=2,
                dtype=dtype,
            ).item()

            total += grad_norm ** 2

    return total ** 0.5


def find_nonfinite_gradients(named_parameters):
    """Locate parameters whose gradients contain inf/nan.

    Returns a list of ``(name, num_nonfinite, numel)`` tuples,
    in parameter-registration order, for every gradient that
    holds at least one non-finite element. An empty list means
    every gradient element is finite (however large).

    Only call this when a cheaper check has already flagged a
    problem - it reduces over every gradient in the model.
    """
    offenders = []

    for name, p in named_parameters:

        if p.grad is None:
            continue

        g = p.grad.detach()

        finite_mask = torch.isfinite(g)

        if not bool(finite_mask.all()):
            offenders.append(
                (
                    name,
                    int((~finite_mask).sum().item()),
                    g.numel(),
                )
            )

    return offenders


def describe_gradients(named_parameters, dtype=torch.float64):
    """Per-parameter gradient report used by the diagnostics.

    Returns a list of dicts with, for each parameter that has a
    gradient:

    - ``finite``:  are all elements finite?
    - ``max_abs``: largest absolute element
    - ``norm32``:  L2 norm computed in the gradients' own fp32
    - ``norm64``:  the same norm accumulated in float64

    ``norm32 == inf`` together with ``finite == True`` is the
    signature of an fp32 reduction overflow on a gradient whose
    elements are all finite but enormous; ``finite == False``
    means there is a genuine inf/nan element.
    """
    report = []

    for name, p in named_parameters:

        if p.grad is None:
            continue

        g = p.grad.detach()

        report.append(
            {
                "name": name,
                "numel": g.numel(),
                "finite": bool(torch.isfinite(g).all()),
                "max_abs": g.abs().max().item(),
                "norm32": g.norm(2).item(),
                "norm64": torch.linalg.vector_norm(
                    g, ord=2, dtype=dtype
                ).item(),
            }
        )

    return report


def state_dict_is_finite(state_dict):
    """Check a model state_dict for inf/nan.

    Returns ``(ok, first_bad_key)``.
    """
    for name, tensor in state_dict.items():

        if not torch.is_tensor(tensor):
            continue

        if not tensor.is_floating_point():
            continue

        if not bool(torch.isfinite(tensor).all()):
            return False, name

    return True, None


def optimizer_state_is_finite(optimizer_state):
    """Check an optimizer state_dict for inf/nan.

    Adam/AdamW moments (``exp_avg``, ``exp_avg_sq``) go NaN as
    soon as a single non-finite gradient is consumed, and a NaN
    second moment poisons every later step even if the weights
    themselves are restored. Returns ``(ok, first_bad_key)``.
    """
    state = optimizer_state.get("state", {})

    for param_id, entries in state.items():

        for key, value in entries.items():

            if not torch.is_tensor(value):
                continue

            if not value.is_floating_point():
                continue

            if not bool(torch.isfinite(value).all()):
                return False, f"state[{param_id}][{key}]"

    return True, None


def resolve_device(requested="auto"):
    """Turn a --device argument into a torch.device.

    ``"auto"`` picks CUDA when it is actually available and
    falls back to CPU otherwise.
    """
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(requested)


def validate(model, val_loader, criterion, normalize_255=True, device=None):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for batch_images, batch_labels in val_loader:

            if normalize_255:
                batch_images = batch_images.float() / 255.0
            else:
                batch_images = batch_images.float()

            if device is not None:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

            output = model(batch_images)

            loss = criterion(
                output,
                batch_labels
            )

            predictions = output.argmax(dim=1)

            total_loss += loss.item() * batch_images.size(0)

            total_correct += (
                predictions == batch_labels
            ).sum().item()

            total_samples += batch_images.size(0)

    avg_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return avg_loss, accuracy
