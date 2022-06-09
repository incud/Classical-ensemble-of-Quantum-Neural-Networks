import numpy as np


class GaussianMixtureClassificationDataset:
    """Create the Gaussian XOR-like Mixture dataset for classification"""
    
    def __init__(self, n, d, padding, epsilon_d, epsilon_padding): 
        """
        Generate the dataset
        :param n: Number of element within the dataset
        :param d: Dimensionality of the feature vector representing the Gaussians
        :param padding: Dimensionality of the feature vector representing 
        the isotropic noise
        :param epsilon_d: Intensity of white noise of the Gaussians 
        :param epsilon_padding: Intensity of isotropic noise
        """
        self.X = []
        self.X_noise = []
        self.y = []
        for i in range(n):
            x_i = self.pick_random_hypercube_point(d, padding)
            y_i = np.prod(x_i[:d])
            noise = self.generate_noise(d, epsilon_d, padding, epsilon_padding)
            self.X.append(x_i)
            self.X_noise.append(x_i + noise)
            self.y.append(y_i)
        
    def save(self, directory): 
        """
        Save to relative path
        :param directory: path of relative directory where the dataset has to 
        be saved
        :return: None
        """
        np.save(f"{directory}/X_wo_noise.npy", self.X) 
        np.save(f"{directory}/X.npy", self.X_noise) 
        np.save(f"{directory}/y.npy", self.y)
    
    @staticmethod
    def pick_random_hypercube_point(d, padding): 
        """
        Pick a random vertex of the d-dimensional hypercube having edge length 
        2 and centered in (0, .., 0)
        :param d: dimensionality of the hypercube
        :param padding: zero padding at the end of the array 
        :return: d-dimensional vector representing the vertex
        """
        point = (2*np.random.randint(0, 2, size=(d,)).astype("float")) -1
        padding = np.zeros(shape=(padding,))
        return np.concatenate((point, padding), axis=0)
    
    @staticmethod
    def generate_noise(d, epsilon_d, padding, epsilon_padding): 
        """
        Generate noise for the feature vector
        :param d:
        :param epsilon_d:
        :param padding:
        :param epsilon_padding:
        :return:
        """
        gaussian_noise = np.random.normal(scale=epsilon_d, size=(d,)) 
        padding_noise = np.random.normal(scale=epsilon_padding, size=(padding,))
        return np.concatenate((gaussian_noise, padding_noise), axis=0)
