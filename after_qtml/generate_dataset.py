import os
import numpy as np
import click


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


@main.command()
def run():
    linear_c(250,  2, 0.1, 1000, 'gaussian', directory='datasets/n250_d02_e01_seed1000')
    linear_c(250,  2, 0.1, 2000, 'gaussian', directory='datasets/n250_d02_e01_seed2000')
    linear_c(250,  2, 0.1, 3000, 'gaussian', directory='datasets/n250_d02_e01_seed3000')
    linear_c(250,  5, 0.1, 1001, 'gaussian', directory='datasets/n250_d05_e01_seed1001')
    linear_c(250,  5, 0.1, 2002, 'gaussian', directory='datasets/n250_d05_e01_seed2002')
    linear_c(250,  5, 0.1, 3003, 'gaussian', directory='datasets/n250_d05_e01_seed3003')
    linear_c(250, 10, 0.1, 1004, 'gaussian', directory='datasets/n250_d10_e01_seed1004')
    linear_c(250, 10, 0.1, 2005, 'gaussian', directory='datasets/n250_d10_e01_seed2005')
    linear_c(250, 10, 0.1, 3006, 'gaussian', directory='datasets/n250_d10_e01_seed3006')
    linear_c(250,  2, 0.5, 1010, 'gaussian', directory='datasets/n250_d02_e05_seed1010')
    linear_c(250,  2, 0.5, 2010, 'gaussian', directory='datasets/n250_d02_e05_seed2010')
    linear_c(250,  2, 0.5, 3010, 'gaussian', directory='datasets/n250_d02_e05_seed3010')
    linear_c(250,  5, 0.5, 1011, 'gaussian', directory='datasets/n250_d05_e05_seed1011')
    linear_c(250,  5, 0.5, 2012, 'gaussian', directory='datasets/n250_d05_e05_seed2012')
    linear_c(250,  5, 0.5, 3013, 'gaussian', directory='datasets/n250_d05_e05_seed3013')
    linear_c(250, 10, 0.5, 1014, 'gaussian', directory='datasets/n250_d10_e05_seed1014')
    linear_c(250, 10, 0.5, 2015, 'gaussian', directory='datasets/n250_d10_e05_seed2015')
    linear_c(250, 10, 0.5, 3016, 'gaussian', directory='datasets/n250_d10_e05_seed3016')


if __name__ == '__main__':
    main()
