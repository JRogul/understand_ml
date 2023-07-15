import numpy as np

def mean_squared_error(y_true, y_pred, root=False):
    """
    Calculate the Mean Squared Error between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The Mean Squared Error.
    """
    if root == True:
         return np.sqrt(np.mean((y_true - y_pred)** 2))
    else:
        return np.mean((y_true - y_pred)** 2) 

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The Mean Absolute Error.
    """

    return np.mean(np.abs(y_true - y_pred))

def accuracy(y_true, y_pred):
    """
    Function to calculate the accuracy of model predictions.\n
    Number of correctly predicted classes / all predictions
    
    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred (np.array): Model's predictions.

    Returns:
    float: The accuracy of the model's predictions.
    """  

    return np.sum(y_true == y_pred) / len(y_true)

def recall(y_true, y_pred):
    """
    Function to calculate the recall for each class.\n
    Recall = TP / (TP + FN)

    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred (np.array): Model's predictions.

    Returns:
    np.array: Recall for each class.
    """

    recalls = []
    for class_idx in range(len(np.unique(y_true))):
        #TP
        num_correct =len(np.where((y_true == class_idx) & (y_pred == np.int64(class_idx)))[0])
        #TP + FN
        total_true =  len(np.where(y_true == class_idx)[0])

        if total_true == 0:
            recalls.append(0)
        else:            
            recalls.append(num_correct / total_true)
    
    return np.array(recalls)

def precision(y_true, y_pred):
    """
    Function to calculate the precision for each class.\n
    Precision = TP / (TP + FP)
    
    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred (np.array): Model's predictions.

    Returns:
    np.array: Precision for each class.
    """

    precisions = []
    for class_idx in range(len(np.unique(y_true))):
        total_preds = np.where(y_pred == np.int64(class_idx))[0] 
        correct_preds = np.where(y_true == class_idx)[0]
        #common elements between 2 arrays
        common = len(np.intersect1d(total_preds, correct_preds)) 
        if len(total_preds) == 0:
            precisions.append(0)
        else:
            precisions.append(common / len(total_preds))
   
    return np.array(precisions)

def f1_score(y_true, y_pred):
    """
    Function to calculate the F1 score for each class.\n
    F1 =  2 * (Precision * Recall) / (Precision + Recall)

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

def print_metrics(model, X_test, y_test):
    """
    Calculate and print the Accuracy, Recall, Precision, and F1 Score 
    of the given model on the test data.

    Parameters:
    model (object): The machine learning model to evaluate.
    X_test (numpy.ndarray): The test feature data.
    y_test (numpy.ndarray): The true labels for the test data.

    Returns:
    None
    """
    
    preds, y_true = model.predict(X_test, y_test)
    print(f'Accuracy: {np.around(accuracy(y_true, preds), 2)*100}%' 
            f'\nRecall: {np.around(recall(y_true, preds), 2)}'
            f'\nPrecision: {np.around(precision(y_true, preds), 2)}'
            f'\nf1score:{np.around(f1_score(y_true, preds), 2)}')
    

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for actual and predicted class labels.

    Parameters:
    y_true (numpy.array): 1D array of actual class labels.
    y_pred (numpy.array): 1D array of predicted class labels.

    Returns:
    numpy.array: The confusion matrix, a 2D array where the i-th row and j-th column 
    entry indicates the number of samples with true label being i-th class 
    and prediced label being j-th class.
    """
    
    conf_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_true))))
    classes = np.unique(np.concatenate([y_true, y_pred]))

    for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                conf_matrix[i, j] = len(np.where((y_true == true_class) & (y_pred == pred_class))[0])
    return conf_matrix
