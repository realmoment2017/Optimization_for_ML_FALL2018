# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 22:22:12 2018

@author: renx0
"""
import time
import numpy as np

class batchGradient():
    def __init__(self, X, y, epochs, lmbda):
        self.X = X
        self.y = y
        self.lmbda = lmbda
        self.w = [np.zeros((self.X.shape[1], 1))]
        self.times = [0.0]
        self.start_time = time.time()
        self.T = epochs
        
    def fit(self):
        radius = 1.0 / ((10.0 * self.lmbda) ** 0.5)
        N, dim = self.X.shape[0], self.X.shape[1]
        print ("Batch gradient descent training")
        w = np.zeros((1, dim))
        W = w
        self.y = self.y.reshape(N, 1)
        for t in range(self.T):        
            eta = 1.0 / (self.lmbda * 10.0)
            q = w - eta * self.lmbda * w
            plane = np.multiply(self.y , np.array(self.X @ np.matrix(w).transpose()))
            mask = (plane < 1)
            labels_tmp = self.y[mask]
            k = np.sum(eta * labels_tmp.reshape(len(labels_tmp), 1) * self.X[np.resize(mask,(len(mask),dim))].reshape(len(labels_tmp), dim), axis=0) / N
            q += k.reshape(1,len(k))
        
            norm_ = np.linalg.norm(q)
            if (norm_ > radius):
                w = q * (radius / norm_)
            else:
                w = q
                
            #W += w           
            self.times.append(time.time() - self.start_time)        
            #t_w = W / (t + 1)
            self.w.append(w.T)
            
        return self.w, self.times