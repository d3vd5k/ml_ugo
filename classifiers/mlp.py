from copy import deepcopy
import os

# Required before CUDA creates a matrix-multiplication handle.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 20, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.001,
    warmup_epochs: int = 100,
    patience: int = 20,
    validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    random_state: int = 42,
    device: str | torch.device | None = None,
    sample_weight: torch.Tensor | None = None,
    hidden_dim: int = 20,
) -> SimpleMLP:
    """Train all supplied rows, stopping only on an explicit real holdout.

    Without validation, run exactly ``epochs`` updates. With validation, restore
    the best holdout-loss checkpoint. ``best_epoch_`` is its epoch count, suitable
    for a final full-data refit. Callers keep the holdout fixed across augmentation
    rounds; this function never splits or substitutes training rows for it.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    selected_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None or str(device) == "auto"
        else torch.device(device)
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_state)
    X = X.to(device=selected_device, dtype=torch.float32)
    y = y.to(device=selected_device, dtype=torch.long)
    weights = None
    if sample_weight is not None:
        weights = torch.as_tensor(sample_weight, device=selected_device, dtype=X.dtype)
        if weights.shape != y.shape or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or float(weights.sum()) <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, aligned and have positive sum.")
        # Mean-one weighting equals the duplicated-data mean loss objective.
        weights = weights / weights.mean()
    classes = torch.unique(y, sorted=True)
    expected_classes = torch.arange(len(classes), device=y.device, dtype=y.dtype)
    if not torch.equal(classes, expected_classes):
        raise ValueError("Classifier labels must be consecutive integers starting at zero.")

    if validation_data is not None:
        X_validation, y_validation = validation_data
        X_validation = X_validation.to(device=selected_device, dtype=X.dtype)
        y_validation = y_validation.to(device=selected_device, dtype=y.dtype)
        if len(y_validation) == 0:
            raise ValueError("validation_data must contain heldout observations.")
        if bool(((y_validation < 0) | (y_validation >= len(classes))).any()):
            raise ValueError("Validation labels must be present in training labels.")

    model = SimpleMLP(input_dim=X.shape[1], hidden_dim=hidden_dim, output_dim=len(classes)).to(selected_device)
    model.training_count_ = len(y)
    model.validation_count_ = 0 if validation_data is None else len(y_validation)
    model.best_epoch_ = epochs
    class_counts = torch.stack([(y == label).sum() for label in classes])
    model.positive_class_ = int(classes[class_counts.argmin()].item())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    best_state = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y) if weights is None else (nn.functional.cross_entropy(logits, y, reduction="none") * weights).mean()
        loss.backward()
        optimizer.step()

        if validation_data is None:
            continue
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(X_validation), y_validation).item())
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            model.best_epoch_ = epoch + 1
            epochs_without_improvement = 0
        elif epoch >= warmup_epochs:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
