from __future__ import division, print_function
import os
import numpy as np
import random as rnd
import time
filepath = os.path.dirname(os.path.abspath(__file__))

class SMO():

    def __init__(self, C=1.0):
        self.C = C
        self.w_ls = []
        self.b_ls = []
        self.time_ls = []
        self.start_time = time.time()
    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        print("SMO training start")
        alpha_prev = np.copy(alpha)
        for j in range(0, n):
            i = self.get_rnd_int(0, n-1, j)
            x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
            k_ij = self.kernel_linear(x_i, x_i) + self.kernel_linear(x_j, x_j) - 2 * self.kernel_linear(x_i, x_j)
            if k_ij == 0:
                continue
            alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
            L, H = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)
            self.w = self.proj(self.calc_w(alpha, y, X),1)
            self.b = self.calc_b(X, y, self.w)
            E_i = self.E(x_i, y_i, self.w, self.b)
            E_j = self.E(x_j, y_j, self.w, self.b)
            alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)
            alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
            self.w_ls.append(self.w)
            self.b_ls.append(self.b)
            self.time_ls.append(time.time() - self.start_time)
        print("SMO training ends....")
        self.b = self.calc_b(X, y, self.w)
        self.w = self.calc_w(alpha, y, X)
    def predict(self, X):
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        return np.mean(y - np.dot(w.T, X.T))
    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
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
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    def proj(self, x, _lambda):
        """Projection of x onto an affine subspace --- 1/np.sqrt(_lambda) ball centered at the origin"""
        if np.linalg.norm(x)>(1/np.sqrt(10*_lambda)):
            x_proj = x/np.linalg.norm(x)*(1/np.sqrt(10*_lambda))
        else:
            x_proj = x
        return x_proj
