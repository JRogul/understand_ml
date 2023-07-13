import numpy as np


def MSELoss(y_true, y_pred):
    """
    Calculate the Mean Squared Error between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The Mean Squared Error.
    """

    return np.mean((y_true - y_pred)** 2) 

def MAELoss(y_true, y_pred):
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
    The softmax function takes an N-dimensional array of real numbers and transforms it 
    into a probability distribution. Each element of the output array is in the range (0, 1), 
    and the total sum of the elements is 1.

    Parameters:
    X (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Probability distribution generated from the input.
    """
 
    return np.exp(X) / np.sum(np.exp(X))

def nll(X, softmaxed=True):
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