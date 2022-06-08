
from dataset import GaussianMixtureClassificationDataset
from classical_rf_model import ClassicalRFModel
import random
import numpy as np
import torch
import torch.nn as nn

random.seed(10)
np.random.seed(10)

n_train = 10
n_test = 10
n = n_train + n_test
d = 2
d_prime = 100
padding = 0
epsilon_d = 0
epsilon_padding = 0


data = GaussianMixtureClassificationDataset(n, d, padding, epsilon_d, epsilon_padding)
x = data.X_noise
y = data.y

model = ClassicalRFModel(d_prime)
model.fit(x,y)



