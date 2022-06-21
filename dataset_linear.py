import numpy as np
import click
import json
import matplotlib.pyplot as plt


class LinearRegressionDataset:
    """Create the Linear Regression dataset"""
    
    def __init__(self, n, d, epsilon):
        """
        Generate the dataset
        :param n: Number of element within the dataset
        :param d: Dimensionality of the feature vector
        """
        self.w = np.random.uniform(low=-1.0, high=1.0, size=(d,))
        self.X = np.random.uniform(low=-0.5, high=0.5, size=(n, d))
        self.y = np.array([self.linear_regression(self.w, xi, epsilon) for xi in self.X])
    
    @staticmethod
    def linear_regression(w, x, noise):
        return np.dot(w, x) + np.random.uniform(low=-noise/2, high=noise/2)


def generate_dataset_process(directory, d, epsilon, n_train, n_test, seed):
    np.random.seed(seed)
    gm_train = LinearRegressionDataset(n=n_train, d=d, epsilon=epsilon)
    np.save(f"{directory}/train_X.npy", gm_train.X)
    np.save(f"{directory}/train_y.npy", gm_train.y)
    gm_test = LinearRegressionDataset(n=n_test, d=d, epsilon=epsilon)
    np.save(f"{directory}/test_X.npy", gm_test.X)
    np.save(f"{directory}/test_y.npy", gm_test.y)
    json.dump({'d': d, 'epsilon': epsilon, 'n_train': n_train, 'n_test': n_test, 'seed': seed}, open(f"{directory}/specs.json", "w"))
    plt.scatter(gm_train.X[:,0], gm_train.X[:,1], c=gm_train.y)
    plt.savefig(f"{directory}/training_set.png")
    plt.close('all')
    plt.scatter(gm_test.X[:,0], gm_test.X[:,1], c=gm_test.y)
    plt.savefig(f"{directory}/testing_set.png")
    plt.close('all')

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
    generate_dataset_process(directory, d, epsilon, n_train, n_test, seed)


@main.command()
@click.option('--directory', type=click.Path(), required=True)
def regenerate_dataset(directory):
    specs = json.load(open(f"{directory}/specs.json"))
    d = int(specs['d'])
    n_train = int(specs['n_train'])
    n_test = int(specs['n_test'])
    seed = int(specs['seed'])
    epsilon = float(specs['epsilon'])
    generate_dataset_process(directory, d, epsilon, n_train, n_test, seed)


if __name__ == '__main__':
    main()

