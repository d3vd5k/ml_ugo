import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
import torch


def imbalanced_accuracy_measure(y_true, y_pred):
    """Compute IAM as defined in the UGO paper."""
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_count = len(labels)
    class_scores = []
    for class_index in range(class_count):
        true_positive = matrix[class_index, class_index]
        false_negative = matrix[class_index].sum() - true_positive
        false_positive = matrix[:, class_index].sum() - true_positive
        denominator = max(matrix[class_index].sum(), matrix[:, class_index].sum())
        class_scores.append(0.0 if denominator == 0 else (true_positive - max(false_negative, false_positive)) / denominator)
    return sum(class_scores) / class_count


def evaluate_and_plot(model, X_test, y_test, show_plot=True, positive_class=None):
    """Report IAM, MCC and positive-class metrics; ``pr_auc`` always means AP.

    Pass the minority label determined from real training labels as
    ``positive_class``. The MLP records that label as ``positive_class_``.
    """
    model.eval()
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    if reference is not None:
        X_test = X_test.to(device=reference.device, dtype=reference.dtype)
    with torch.no_grad():
        probabilities = torch.softmax(model(X_test), dim=1)
        preds = torch.argmax(probabilities, dim=1).cpu().numpy()

    y_test_np = y_test.detach().cpu().numpy()
    if positive_class is None:
        positive_class = getattr(model, "positive_class_", None)
    if positive_class is None:
        raise ValueError("Provide a positive_class determined from real training labels.")
    positive_class = int(positive_class)
    if not 0 <= positive_class < probabilities.shape[1]:
        raise ValueError("positive_class must identify a model probability column.")
    binary_targets = (y_test_np == positive_class).astype(int)
    binary_predictions = (preds == positive_class).astype(int)
    f1 = f1_score(binary_targets, binary_predictions, zero_division=0)
    mcc = matthews_corrcoef(y_test_np, preds)
    iam = imbalanced_accuracy_measure(y_test_np, preds)
    accuracy = accuracy_score(y_test_np, preds)
    precision = precision_score(binary_targets, binary_predictions, zero_division=0)
    recall = recall_score(binary_targets, binary_predictions, zero_division=0)
    positive_probabilities = probabilities[:, positive_class].cpu().numpy()
    pr_auc = average_precision_score(binary_targets, positive_probabilities)
    print(
        f"IAM: {iam:.3f} | MCC: {mcc:.3f} | F1 (positive class {positive_class}): {f1:.3f} | "
        f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | AP: {pr_auc:.3f}"
    )

    if show_plot:
        precision_arr, recall_arr, _ = precision_recall_curve(binary_targets, positive_probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(recall_arr, precision_arr, color='blue', lw=2, label=f'PR Curve (AP = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (Average Precision)')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.show()

    return {
        'iam': iam,
        'mcc': mcc,
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc,
    }
