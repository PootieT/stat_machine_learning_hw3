import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  dummy, K = np.shape(theta)

  for i in range(m):
    temp = np.dot(theta.T, X[i,:])
    temp = np.exp(temp - np.ones_like(temp) * np.amax(temp))
    temp_sum = 0
    for j in range(len(temp)):
      temp_sum += temp[j]
    J += -1.0 / float(m) * np.log(temp[y[i]] / temp_sum)
    for k in range(K):
      grad[:,k] += -1.0 / float(m) * ( int(y[i]==k) - (temp[k] / temp_sum) )  * X[i,:] 

  for j in range(dim):
    for k in range(1, K):
      J += reg * theta[j,k]**2 / (2.0 * m)
  
  for k in range(K):
    grad[:,k] += reg / float(m) * theta[:,k]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  dummy, K = np.shape(theta)
  temp = np.matmul(X, theta)
  #print(np.shape(temp), np.shape(np.amax(temp,axis=1)))
  #temp = np.exp(temp - np.tile(np.amax(temp, axis=1), (K,1)).T)
  temp = np.exp((temp.T - np.amax(temp, axis=1)).T)
  sums = np.sum(temp, axis=1)
  probs = (temp.T / sums).T
  J += -1 / float(m) * np.sum(np.log(probs[range(m), y])) 
  J += reg / (2.0*m) * np.sum(np.square(theta[:,1:]))

  indicator = np.zeros(np.shape(probs))
  indicator[range(m), y] = 1.0
  grad += -1.0 / float(m) * ( np.matmul(X.T,(indicator - probs)) ) + reg / float(m) * theta
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
