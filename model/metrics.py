import torch

def accuracy(y_pred, y_true):
    """
    Calculate accuracy for binary classification.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0 or 1)
        
    Returns:
        Accuracy score
    """
    y_pred_tag = torch.round(y_pred)
    correct = (y_pred_tag == y_true).float().sum()
    return correct / y_true.shape[0]

def precision(y_pred, y_true):
    """
    Calculate precision for binary classification.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0 or 1)
        
    Returns:
        Precision score
    """
    y_pred_tag = torch.round(y_pred)
    true_positives = ((y_pred_tag == 1) & (y_true == 1)).float().sum()
    predicted_positives = (y_pred_tag == 1).float().sum()
    return true_positives / (predicted_positives + 1e-7)

def recall(y_pred, y_true):
    """
    Calculate recall for binary classification.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0 or 1)
        
    Returns:
        Recall score
    """
    y_pred_tag = torch.round(y_pred)
    true_positives = ((y_pred_tag == 1) & (y_true == 1)).float().sum()
    actual_positives = (y_true == 1).float().sum()
    return true_positives / (actual_positives + 1e-7)

def f1_score(y_pred, y_true):
    """
    Calculate F1 score for binary classification.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0 or 1)
        
    Returns:
        F1 score
    """
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return 2 * (prec * rec) / (prec + rec + 1e-7)

def get_all_metrics(y_pred, y_true):
    """
    Calculate all metrics for binary classification.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0 or 1)
        
    Returns:
        Dictionary with accuracy, precision, recall and f1 score
    """
    return {
        'accuracy': accuracy(y_pred, y_true).item(),
        'precision': precision(y_pred, y_true).item(),
        'recall': recall(y_pred, y_true).item(),
        'f1': f1_score(y_pred, y_true).item()
    }