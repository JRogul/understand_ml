import numpy as np

#softmax
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))

#Negative log likelihood
def nll(X, softmaxed=True):
    if softmaxed== False:
        #first softmax, then -log(x)
        return np.sum(- np.log10(np.exp(X) / np.sum(np.exp(X))))
    else:
        #-log(x)
        return np.sum(- np.log10(X))
    
#Multiclass SVM loss
def multi_class_svm_loss(X, class_index, margin=1):
    loss = 0 
    for x in X:
        loss += max(0, x- X[class_index] + margin)

    #substracting -1 for the index == correct class
    return loss - 1