########## Classical Models ##########

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
    
    
def output(x, theta, w):
    phi = np.dot(theta,x)
    result = np.dot(w,phi).T
    return result

def E(weights, params): 
    """
    Calculate Mean Squared Error
    :param y_pred: vector of predicted targets
    :param y: vector of true targets
    :return: the MSE cost
    """
    x = params[0]
    y_true = params[1]
    theta = params[2]
    y_pred = output(x, theta, weights).reshape(-1,1)
    P = y_true.shape[0]
    return np.asscalar((1/(2*P))*np.dot((y_pred-y_true).T,(y_pred-y_true)))
    
class ClassicalRFModel:
    def __init__(self, d_prime):
        self.d_prime = d_prime
        self.theta = None
        self.w = None
    

    def fit(self, x, y):
        x = np.array(x).T
        y = np.array(y).reshape(-1,1)
        self.theta = np.random.normal(size=(self.d_prime, x.shape[0]))
        print(self.theta.shape)
        self.w = np.random.rand(1, self.d_prime)
        params=[x,y,self.theta,self.w]
        res = minimize(E, x0=self.w, args=params, 
                 method='BFGS',  options={'disp':True})
        self.w = res.x
        print('\nEstimated training error: ', E(self.w,params))
        return self
    
    def predict(self, x):
        return output(x, self.theta, self.w)
    