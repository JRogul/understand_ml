import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function.

    The sigmoid function maps any real value to a value between 0 and 1 using the formula:
    f(x) = 1 / (1 + exp(-x))

    Parameters:
        x (float): Input value.

    Returns:
        float: Output value between 0 and 1.
    """

    return 1 / (1 + np.exp(x))

def relu(x, leaky=False):
    """
    ReLU (Rectified Linear Unit) activation function.

    The ReLU function returns 0 for any negative input value and returns the input value itself for any positive input value using the formula:
    f(x) = max(0, x)

    Parameters:
        x (float): Input value.
        leaky (bool): Whether to use leaky ReLU variant with a small slope for negative input values.

    Returns:
        float: Output value, 0 for negative input, and x for positive input.
    """

    if leaky == False:
        return max(0, x)
    else:
        return max(0.01 * x, x)
    
def tanh(x):
    """
    Hyperbolic Tangent (tanh) activation function.

    The tanh function maps the input values between -1 and 1 using the formula:
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Parameters:
        x (float): Input value.

    Returns:
        float: Output value between -1 and 1.
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))