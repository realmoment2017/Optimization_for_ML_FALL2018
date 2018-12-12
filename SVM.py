"""
    Author: Lasse Regin Nielsen
"""

from __future__ import division, print_function
import os
import numpy as np
import random as rnd
import time
filepath = os.path.dirname(os.path.abspath(__file__))

class SVM():
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm for training.
    """
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.w_ls = []
        self.b_ls = []
        self.time_ls = []
        self.start_time = time.time()
    def fit(self, X, y):
        # Initialization
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel_type]
        count = 0
        print("SMO training start")
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, 3010):
                # tt = time.time()
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                # print("before kernel timestamp {}".format(time.time() - self.start_time))
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                # print("after kernel timestamp {}".format(time.time() - self.start_time))
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                # print("before compute_L_H timestamp {}".format(time.time() - self.start_time))
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)
                # print("after compute_L_H timestamp {}".format(time.time() - self.start_time))

                # Compute model parameters
                # print("before w {}".format(time.time() - self.start_time))
                self.w = self.proj(self.calc_w(alpha, y, X),1)
                # print("after w timestamp {}".format(time.time() - self.start_time))
                self.b = self.calc_b(X, y, self.w)
                # print("after b timestamp {}".format(time.time() - self.start_time))

                # Compute E_i, E_j
                # print("before E {}".format(time.time() - self.start_time))
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)
                # print("after E timestamp {}".format(time.time() - self.start_time))
                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
                
                self.w_ls.append(self.w)
                self.b_ls.append(self.b)
                self.time_ls.append(time.time() - self.start_time)
                # print(time.time() - self.start_time)
            # Check convergence
            # print("finish one iter")
            print("SMO training ends....")
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                print("using time : {}".format(time.time() - self.start_time))
                return
        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count
    def predict(self, X):
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def calc_w(self, alpha, y, X):
        # print("X dims {}".format(X.shape))
        # print("alpha dims {}".format(alpha.shape))
        # print("y dims {}".format(y.shape))
        return np.dot(X.T, np.multiply(alpha,y))
    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = rnd.randint(a,b)
            cnt=cnt+1
        return i
    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)
    def proj(self, x, _lambda):
        """Projection of x onto an affine subspace --- 1/np.sqrt(_lambda) ball centered at the origin"""
        if np.linalg.norm(x)>(1/np.sqrt(_lambda)):
            x_proj = x/np.linalg.norm(x)*(1/np.sqrt(_lambda))
        else:
            x_proj = x
        return x_proj
