import torch
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_labels = (y_pred > threshold).astype(int)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    recall_pos = tp / (tp + fn)  # Recall for positive class (Sensitivity)
    recall_neg = tn / (tn + fp)  # Recall for negative class (Specificity)
    balanced_accuracy = (recall_pos + recall_neg) / 2

    return balanced_accuracy, precision, recall, f1


def evaluate(loader, model, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index)

            target = batch.y.float().unsqueeze(1)

            loss = criterion(out, target)
            total_loss += loss.item()

            probs = torch.sigmoid(out)
            preds = (probs >= threshold).long()

            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean((np.array(all_preds) > threshold) == np.array(all_labels))
    balanced_accuracy, precision, recall, f1 = calculate_metrics(np.array(all_labels), np.array(all_preds), threshold=threshold)
    return avg_loss, accuracy, balanced_accuracy, precision, recall, f1