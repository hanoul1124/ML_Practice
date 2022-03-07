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
    # https://mainpower4309.tistory.com/29
    scores = X.dot(W)
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        f = scores[i]
        softmax = np.exp(f) / np.sum(np.exp(f))
        loss += -np.log(softmax[y[i]]) # NLL Loss
        # Weight에 대한 gradient는 Loss식을 정리한 후 편미분 시 다음과 같이 나타난다
        for j in range(num_class):
            dW[:, j] += X[i] * softmax[j]
        dW[:, y[i]] -= X[i]
    
    # average    
    loss /= num_train
    dW /= num_train
    
    # Regularization
    loss += reg * np.sum(W*W) # regularization(L2 Norm)
    dW += reg * 2 * W
        
        
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    num_train = X.shape[0]
    
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, np.newaxis]
    loss += np.mean(-np.log(p[np.arange(num_train)], y))
    loss += reg * np.sum(W*W)
    
    softmax[np.arange(num_train), y] -= 1
    softmax /= num_train
    dW = X.T.dot(softmax)
    dW += reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
