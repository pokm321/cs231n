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

    C = W.shape[1] # 클래스 갯수
    N = X.shape[0] # 이미지 갯수

    for i in range(N):
      scores = X[i] @ W
      scores_exp = np.exp(scores - scores.max())
      # "수치적 안정성"을 위해 일정한 값 scores.max()를 빼서 Loss값에 변화는 주지 않고 e지수값들을 작아지게 함.
      # e^(s-n) = (e^s)*(e^-n)
      probability = scores_exp[y[i]]/np.sum(scores_exp)
      loss -= np.log(probability)
      dW += X[i][:,np.newaxis] * (scores_exp) * (-1 * probability / np.sum(scores_exp)) * (-1/probability)
      dW[:,y[i]] += X[i] * (scores_exp[y[i]]) * (1 / np.sum(scores_exp)) * (-1/probability)

    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + reg * 2 * W

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    C = W.shape[1] # 클래스 갯수
    N = X.shape[0] # 이미지 갯수

    scores = X @ W
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probability = scores_exp[range(N),y]/np.sum(scores_exp, axis=1)
    loss -= np.sum(np.log(probability))
    dW += X.T @ (scores_exp / np.sum(scores_exp, axis=1, keepdims=True))
    dW -= X.T @ (np.arange(C) == y[:, None])

    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
