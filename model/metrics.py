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

def far(y_pred, y_true):
    """
    Calculate False Acceptance Rate (FAR) for biometric verification.
    FAR is the rate at which unauthorized users are accepted as genuine.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0 or 1)
        
    Returns:
        False Acceptance Rate
    """
    y_pred_tag = torch.round(y_pred)
    
    # False positives (system accepts impostor)
    false_positives = ((y_pred_tag == 1) & (y_true == 0)).float().sum()
    
    # Total impostor attempts
    impostor_attempts = (y_true == 0).float().sum()
    
    # Avoid division by zero
    return false_positives / (impostor_attempts + 1e-7)

def frr(y_pred, y_true):
    """
    Calculate False Rejection Rate (FRR) for biometric verification.
    FRR is the rate at which genuine users are rejected as impostors.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0 or 1)
        
    Returns:
        False Rejection Rate
    """
    y_pred_tag = torch.round(y_pred)
    
    # False negatives (system rejects genuine user)
    false_negatives = ((y_pred_tag == 0) & (y_true == 1)).float().sum()
    
    # Total genuine attempts
    genuine_attempts = (y_true == 1).float().sum()
    
    # Avoid division by zero
    return false_negatives / (genuine_attempts + 1e-7)

def apcer(y_pred, y_true):
    """
    Attack Presentation Classification Error Rate (APCER/FerrFake).
    Rate of misclassified fake fingerprints (attack samples incorrectly classified as genuine).
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=fake, 1=genuine)
        
    Returns:
        APCER score
    """
    y_pred_tag = torch.round(y_pred)
    
    # Fake samples classified as genuine (false positives for fake samples)
    misclassified_fakes = ((y_pred_tag == 1) & (y_true == 0)).float().sum()
    
    # Total fake samples
    total_fakes = (y_true == 0).float().sum()
    
    return misclassified_fakes / (total_fakes + 1e-7)

def bpcer(y_pred, y_true):
    """
    Bona fide Presentation Classification Error Rate (BPCER/FerrLive).
    Rate of misclassified live fingerprints (genuine samples incorrectly classified as fake).
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=fake, 1=genuine)
        
    Returns:
        BPCER score
    """
    y_pred_tag = torch.round(y_pred)
    
    # Genuine samples classified as fake (false negatives)
    misclassified_live = ((y_pred_tag == 0) & (y_true == 1)).float().sum()
    
    # Total genuine samples
    total_live = (y_true == 1).float().sum()
    
    return misclassified_live / (total_live + 1e-7)

def fnmr(y_pred, y_true):
    """
    False Non-Match Rate (FNMR).
    Rate at which genuine comparisons are incorrectly classified as impostor comparisons.
    Equivalent to 100% - Genuine Acceptance Rate (GAR).
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=impostor, 1=genuine)
        
    Returns:
        FNMR score
    """
    y_pred_tag = torch.round(y_pred)
    
    # Genuine comparisons classified as impostor (false negatives)
    false_non_matches = ((y_pred_tag == 0) & (y_true == 1)).float().sum()
    
    # Total genuine comparisons
    total_genuine = (y_true == 1).float().sum()
    
    return false_non_matches / (total_genuine + 1e-7)

def gar(y_pred, y_true):
    """
    Genuine Acceptance Rate (GAR).
    Rate at which genuine comparisons are correctly classified.
    Equivalent to 100% - FNMR.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=impostor, 1=genuine)
        
    Returns:
        GAR score
    """
    # GAR = 1 - FNMR
    return 1.0 - fnmr(y_pred, y_true)

def fmr(y_pred, y_true):
    """
    False Match Rate (FMR).
    Equivalent to False Accept Rate (FAR) in verification scenarios.
    Rate at which impostor comparisons are incorrectly classified as genuine.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=impostor, 1=genuine)
        
    Returns:
        FMR score
    """
    # Same implementation as FAR
    return far(y_pred, y_true)

def iapmr(y_pred, y_true):
    """
    Impostor Attack Presentation Match Rate (IAPMR).
    Rate at which attack presentations succeed in being accepted.
    Related to 100-IMI_accuracy or 100-SGAR.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=impostor/fake, 1=genuine)
        
    Returns:
        IAPMR score
    """
    y_pred_tag = torch.round(y_pred)
    
    # Attack presentations accepted (false positives)
    successful_attacks = ((y_pred_tag == 1) & (y_true == 0)).float().sum()
    
    # Total attack presentations
    total_attacks = (y_true == 0).float().sum()
    
    return successful_attacks / (total_attacks + 1e-7)

def img_accuracy(y_pred, y_true):
    """
    Image-level accuracy (IMG_accuracy).
    Overall accuracy for distinguishing live vs fake presentations.
    Related to FNMR as: FNMR = 100 - IMG_accuracy
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=fake, 1=genuine)
        
    Returns:
        IMG_accuracy score
    """
    return accuracy(y_pred, y_true)

def sgar(y_pred, y_true):
    """
    Spoof/Genuine Accept Rate (SGAR).
    The rate at which the system correctly accepts genuine presentations
    and rejects spoof/fake presentations.
    
    Args:
        y_pred: Predicted values (0-1)
        y_true: Ground truth values (0=fake, 1=genuine)
        
    Returns:
        SGAR score
    """
    y_pred_tag = torch.round(y_pred)
    
    # Correctly classified samples (both genuine and fake)
    correct = (y_pred_tag == y_true).float().sum()
    
    # Total samples
    total = y_true.shape[0]
    
    return correct / total


def get_all_metrics(y_pred, y_true):

    return {
        # Basic metrics
        'accuracy': accuracy(y_pred, y_true).item(),
        'precision': precision(y_pred, y_true).item(),
        'recall': recall(y_pred, y_true).item(),
        'f1': f1_score(y_pred, y_true).item(),
        
        # Verification metrics
        'far': far(y_pred, y_true).item(),
        'frr': frr(y_pred, y_true).item(),
        'fmr': fmr(y_pred, y_true).item(),
        'fnmr': fnmr(y_pred, y_true).item(),
        'gar': gar(y_pred, y_true).item(),
        
        # Presentation attack detection metrics
        'apcer': apcer(y_pred, y_true).item(),
        'bpcer': bpcer(y_pred, y_true).item(),
        'iapmr': iapmr(y_pred, y_true).item(),
        'img_accuracy': img_accuracy(y_pred, y_true).item(),
        'sgar': sgar(y_pred, y_true).item()
    }