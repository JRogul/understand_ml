import numpy as np


def mean_squared_error_loss(y_true, y_pred):
    """
    Calculate the Mean Squared Error between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The Mean Squared Error.
    """

    return np.mean((y_true - y_pred)** 2) 

def mean_absolute_error_loss(y_true, y_pred):
    """
    Calculate the Mean Absolute Error between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The Mean Absolute Error.
    """

    return np.mean(np.abs(y_true - y_pred))

def softmax(X):
    """
    Converts an N-dimensional array to a probability distribution using the softmax function.
    
    Parameters:
    X (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array as a probability distribution.
    """
 
    if len(X.shape) == 1:
        return np.exp(X) / np.sum(np.exp(X))
    elif len(X.shape) == 2:
        return np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)

def negative_log_likelihood(X, softmaxed=True):
    """
    The negative log likelihood (NLL) function measures the dissimilarity between an estimated 
    probability distribution (from the softmax function) and the true distribution.

    Parameters:
    X (numpy.ndarray): Input array.
    softmaxed (bool): Indicates whether the input array has already been softmaxed. 
                      Defaults to True.

    Returns:
    float: The negative log likelihood of the input.
    """

    if softmaxed== False:
        #first softmax, then -log(x)
        return np.sum(- np.log10(softmax(X)))
    else:
        #-log(x)
        return np.sum(- np.log10(X))
    
def multi_class_svm_loss(X, class_index, margin=1):
    
    """
    The multiclass SVM loss function measures the quality of a model's prediction 
    of the correct class for given data. 

    Parameters:
    X (numpy.ndarray): Input array.
    class_index (int): Index of the correct class in the input array.
    margin (float): Margin size for the loss calculation. Defaults to 1.

    Returns:
    float: The multiclass SVM loss of the input.
    """

    loss = 0 
    for x in X:
        loss += max(0, x- X[class_index] + margin)

    #substracting -1 for the index == correct class
    return loss - 1

def cross_entropy_loss(y_pred, y_test):
    """
    Computes the cross-entropy loss between predicted and true labels.

    Parameters:
    y_pred (np.ndarray): Predicted probabilities from the model.
    y_test (np.ndarray): True labels.

    Returns:
    float: The cross-entropy loss.
    """

    soft = softmax(y_pred)
    column_indicies = y_test
    row_indicies = range(len(soft))
    loss = -1 * np.sum(np.log(soft[row_indicies, column_indicies])) / len(soft)
    return loss