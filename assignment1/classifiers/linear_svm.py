from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]   # number of columns for the W matrix? 
    num_train = X.shape[0]      
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]   # getting the label class for the specific X example 
        for j in range(num_classes):
            if j == y[i]: # If j is one of the classes in y[i] i.e. y is the array of classes i.e. [3, 0, 1,...,9] 3 refers to label for image in X[0] and so on..
                continue
            margin = (scores[j] - correct_class_score+1) # we get the scores for the j
            # class in the dataset i.e. 10 and calculate the margin across all classes
            
            # margin = np.max(0, (scores[j] - correct_class_score + 1)) # note delta = 1
            if margin > 0:
                loss += margin
                # the derivative of the function w.r.t weight is X
                
                dW[:, j] += X[i,:] # i refers to the row and : refers to the columns
                dW[:, y[i]] -= X[i,:]   # y[i] the y label for i-th example. 
                # 
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # dW = 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    yi_scores = scores[np.arange(scores.shape[0]), y]
    margins = np.maximum(0, (scores - np.matrix(yi_scores).T + 1))
    margins[np.arange(num_train), y] = 0
    
    loss = np.mean(np.sum(margins, axis=1))
    loss += 0.5 * reg * np.sum(W*W)
    
    binary = margins
    binary[margins > 0] = 1
    
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum.T
    
    dW = np.dot(X.T, binary)
    
    dW /= num_train
    dW += reg * W
    correct_scores = scores[y]
    margins = (scores - correct_scores + 1)
    # probably need to do a for loop here that goes through the margins?
    
    # if margins > 0:
    #     loss += margins
    #     dW[:, ] += X
    #     dW[:, correct_scores] -= X 
    
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
