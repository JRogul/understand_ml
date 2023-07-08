import numpy as np

def accuracy(y_true, y_pred):
    """
    Function to calculate the accuracy of model predictions.

    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred (np.array): Model's predictions.

    Returns:
    float: The accuracy of the model's predictions.
    """  

    return np.sum(y_true == y_pred) / len(y_true)

def recall(y_true, y_pred):
    """
    Function to calculate the recall for each class.

    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred (np.array): Model's predictions.

    Returns:
    np.array: Recall for each class.
    """

    recalls = []
    for class_idx in range(len(np.unique(y_true))):
        num_correct =len(np.where((y_true == class_idx) & (y_pred == class_idx))[0])
        total_true =  len(np.where(y_true == class_idx)[0])

        if total_true == 0:
            recalls.append(0)
        else:            
            recalls.append(num_correct / total_true)
    
    return np.array(recalls)

def precision(y_true, y_pred):
    """
    Function to calculate the precision for each class.

    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred (np.array): Model's predictions.

    Returns:
    np.array: Precision for each class.
    """

    precisions = []
    for class_idx in range(len(np.unique(y_true))):
        total_preds = np.where(y_pred == class_idx)[0] 
        correct_preds = np.where(y_true == class_idx)[0]
        #common elements between 2 arrays
        common = len(np.intersect1d(total_preds, correct_preds)) 
        if total_preds == 0:
            precisions.append(0)
        else:
            precisions.append(common / len(total_preds))
   
    return np.array(precisions)

def f1_score(y_true, y_pred):
    """
    Function to calculate the F1 score for each class.

    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred (np.array): Model's predictions.

    Returns:
    np.array: F1 score for each class.
    """

    precisions = precision(y_true, y_pred) 
    recalls  = recall(y_true, y_pred)
    f1_scores = []
    for class_idx in range(len(np.unique(y_true))):
        f1_score = 2* (precisions[class_idx] * recalls[class_idx]) \
                    / (precisions[class_idx] + recalls[class_idx] + 1e-8) 
        f1_scores.append(f1_score)
    return np.array(f1_scores)