import numpy as np
import click
import json
import matplotlib.pyplot as plt


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
            x_i = self.pick_random_hypercube_point(d, padding) / 2
            y_i = np.prod(x_i[:d])
            noise = self.generate_uniform_noise(d, epsilon_d, padding, epsilon_padding)
            self.X.append(x_i)
            self.X_noise.append(x_i + noise)
            self.y.append(y_i)

        self.X = np.array(self.X)
        self.X_noise = np.array(self.X_noise)
        self.y = np.array(self.y)
        
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
    def generate_uniform_noise(d, epsilon_d, padding, epsilon_padding):
        """
        Generate noise for the feature vector
        :param d:
        :param epsilon_d:
        :param padding:
        :param epsilon_padding:
        :return:
        """
        uniform_noise = np.random.uniform(low=-epsilon_d/2, high=epsilon_d/2, size=(d,))
        padding_noise = np.random.uniform(low=-epsilon_padding/2, high=epsilon_padding/2, size=(padding,))
        return np.concatenate((uniform_noise, padding_noise), axis=0)


@click.group()
def main():
    pass


@main.command()
@click.option('--directory', type=click.Path(), required=True)
@click.option('--d', type=int, required=True)
@click.option('--epsilon', type=float, required=True)
@click.option('--n-train', type=int, required=True)
@click.option('--n-test', type=int, required=True)
@click.option('--seed', type=int, required=True)
def generate_dataset(directory, d, epsilon, n_train, n_test, seed):
    np.random.seed(seed)
    gm_train = GaussianMixtureClassificationDataset(n=n_train, d=d, padding=0, epsilon_d=epsilon, epsilon_padding=0.0)
    np.save(f"{directory}/train_X_wo_noise.npy", gm_train.X)
    np.save(f"{directory}/train_X.npy", gm_train.X_noise)
    np.save(f"{directory}/train_y.npy", gm_train.y)
    gm_test = GaussianMixtureClassificationDataset(n=n_train, d=d, padding=0, epsilon_d=epsilon, epsilon_padding=0.0)
    np.save(f"{directory}/test_X_wo_noise.npy", gm_test.X)
    np.save(f"{directory}/test_X.npy", gm_test.X_noise)
    np.save(f"{directory}/test_y.npy", gm_test.y)
    json.dump({'d': d, 'epsilon': epsilon, 'n_train': n_train, 'n_test': n_test, 'seed': seed}, open(f"{directory}/specs.json", "w"))
    plt.scatter(gm_train.X_noise[:,0], gm_train.X_noise[:,1], c=gm_train.y)
    plt.savefig(f"{directory}/training_set.png")
    plt.close('all')
    plt.scatter(gm_test.X_noise[:,0], gm_test.X_noise[:,1], c=gm_test.y)
    plt.savefig(f"{directory}/testing_set.png")
    plt.close('all')


if __name__ == '__main__':
    main()

