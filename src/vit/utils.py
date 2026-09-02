import torch


def get_gradient_norm(parameters):
    total = 0.0

    for p in parameters:
        if p.grad is not None:
            grad_norm = p.grad.detach().norm(2).item()
            total += grad_norm ** 2

    return total ** 0.5


def validate(model, val_loader, criterion, normalize_255=True):

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
