from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = np.zeros(W.shape)
    n_class = W.shape[1]
    N, D = X.shape
    loss = 0.0
    for i in range(N):
        score = np.dot(X[i], W)
        sum_s = np.sum(np.exp(score))
        # the correct class of the i-th example is exponentiated
        corr_s = np.exp(score[y[i]])
        # loss by getting -log:
        loss = -1 * np.log(corr_s/sum_s)
        for j in range(n_class):
            dW[:, j] += X[i, :]
            dW[:, y[i]] -= X[i, :]
            
    loss /= N
    dW /= N
    loss += reg * np.sum(W*W)
    dW += reg*W
    # find the gradient:
    # the gradient would still be the same where the 
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # no loops loss:
    scores = np.dot(X, W)
    # exponentiate the loss:
    n_scores = np.sum(np.exp(scores))
    c_corr = np.exp(scores[y[i]])
    # exponentiate
    exp = -1 * np.log(c_corr/n_scores)
    corr_prob = exp[range(num_train), y]
    loss = np.sum(corr_prob)
    loss += 0.5 * reg * np.sum(W**2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dscores = prob_scores
    dscores[range(num_train), y] -= 1
    dW = np.dot(X.T, dscores)
    dW /= num_train
    dW += reg * W
    return loss, dW
