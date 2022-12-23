import os
import numpy as np
import click
from sklearn.datasets import load_diabetes, fetch_california_housing, load_iris


@click.group()
def main():
    pass


@main.command()
@click.option('--n', type=int, required=True)
@click.option('--d', type=int, required=True)
@click.option('--e', type=float, required=True)
@click.option('--seed', type=int, required=True)
@click.option('--noise', type=click.Choice('gaussian', 'uniform'), required=True)
def linear(n, d, e, seed, noise):
    linear_c(n, d, e, seed, noise)


def linear_c(n, d, e, seed, noise, directory='.'):
    np.random.seed(seed)
    w = np.random.uniform(low=-1.0, high=1.0, size=(d, ))
    assert len(w.ravel()) == d, f"Vector w must have dimensionality d={d}"
    X = np.random.uniform(low=-1.0, high=1.0, size=(n, d))
    assert X.shape == (n, d)
    y_no_noise = np.array([np.dot(w, x) for x in X])
    assert y_no_noise.shape == (n,)
    if noise == 'gaussian':
        epsilon = np.random.normal(scale=e, size=(n,))
    else:
        epsilon = (e/2) * np.random.uniform(size=(n,))
    assert epsilon.shape == (n,)
    y = y_no_noise + epsilon
    os.makedirs(directory, exist_ok=True)
    np.save(f"{directory}/X.npy", X)
    np.save(f"{directory}/w.npy", w)
    np.save(f"{directory}/y_no_noise.npy", y_no_noise)
    np.save(f"{directory}/y.npy", y)
    # print('y_no_noise',y_no_noise)
    # print('max y_no_noise',max(y_no_noise))
    # print()    
    # print('epsilon',epsilon)
    # print('max epsilon',max(epsilon))
    # import matplotlib.pyplot as plt
    # direction = 0
    # x1 = X[:, direction]
    # plt.scatter(x1, y, label='Sampled points')
    # plt.plot(x1, x1 * w[direction], '--', label=f'Slope in direction {direction}')
    # plt.legend()
    # plt.ylim(-3, 3)
    # plt.title("Plot of (noisy) data points")
    # plt.xlabel("The x label")
    # plt.ylabel("The y label")
    # plt.show()
    # plt.close()



@main.command()
@click.option('--n', type=int, required=True)
@click.option('--e', type=float, required=True)
@click.option('--seed', type=int, required=True)
@click.option('--noise', type=click.Choice('gaussian', 'uniform'), required=True)
def sin_dataset(n, e, seed, noise):
    linear_c(n, e, seed, noise)


def sin_dataset_c(n, e, seed, noise, directory='.'):
    np.random.seed(seed)
    w = np.pi
    X = np.random.uniform(low=-1.0, high=1.0, size=(n, 1))
    assert X.shape == (n, 1)
    y_no_noise = np.array([np.sin(x*w) for x in X]).reshape(-1,)
    assert y_no_noise.shape == (n,)
    if noise == 'gaussian':
        epsilon = np.random.normal(scale=e, size=(n,))
    else:
        epsilon = (e/2) * np.random.uniform(size=(n,))
    assert epsilon.shape == (n,)
    y = y_no_noise + epsilon
    os.makedirs(directory, exist_ok=True)
    np.save(f"{directory}/X.npy", X)
    np.save(f"{directory}/w.npy", w)
    np.save(f"{directory}/y_no_noise.npy", y_no_noise)
    np.save(f"{directory}/y.npy", y)


@main.command()
def diabete_dataset():
    diabete_dataset_c()


def diabete_dataset_c(scaled=False):
    X,y = load_diabetes(return_X_y=True,scaled=scaled)
    directory = 'datasets/diabete'
    os.makedirs(directory, exist_ok=True)
    np.save(f"{directory}/X.npy", X)
    np.save(f"{directory}/y.npy", y)
    
    
    
@main.command()
def housing_dataset():
    housing_dataset_c()


def housing_dataset_c():
    directory = 'datasets/housing'
    os.makedirs(directory, exist_ok=True)
    X,y = fetch_california_housing(data_home=directory,return_X_y=True)
    # ADD SOME PREPROCESSING HERE?
    np.save(f"{directory}/X.npy", X)
    np.save(f"{directory}/y.npy", y)
    
    
@main.command()
def iris_dataset():
    iris_dataset_c()


def iris_dataset_c():
    directory = 'datasets/iris'
    os.makedirs(directory, exist_ok=True)
    X,y = load_iris(return_X_y=True)
    # ADD SOME PREPROCESSING HERE?
    np.save(f"{directory}/X.npy", X)
    np.save(f"{directory}/y.npy", y)


@main.command()
def run():
    # linear_c(1000,  2, 0.1, 1000, 'gaussian', directory='datasets/linear/n1000_d02_e01_seed1000')

    
    # sin_dataset_c(250, 0.1, 1000, 'gaussian', directory='datasets/sin/n250_e01_seed1000')
    # sin_dataset_c(250, 0.5, 1000, 'gaussian', directory='datasets/sin/n250_e05_seed2000')
    
    #diabete_dataset_c()
    housing_dataset_c()
    iris_dataset_c()

if __name__ == '__main__':
    main()
