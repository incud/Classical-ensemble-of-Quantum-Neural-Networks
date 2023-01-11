import os
import numpy as np
import click
import pathlib
import matplotlib.pyplot as plt


@click.group()
def main():
    pass


@main.command()
@click.option('--dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--d', type=int, required=False)
def plot_dataset(dataset,d):
    plot_dataset_c(dataset, d)
    
    
def plot_dataset_c(dataset, d=None):
    dataset = pathlib.Path(dataset)
    X = np.load(dataset / "X.npy")
    y = np.load(dataset / "y.npy")
    w = np.load(dataset / "w.npy")
    direction = d if d is not None else 0
    x1 = X[:, direction]
    plt.scatter(x1, y, label='Sampled points')
    plt.plot(x1, x1 * w[direction], '--', label=f'Slope in direction {direction}')
    plt.legend()
    plt.ylim(-3, 3)
    plt.title("Plot of (noisy) data points")
    plt.xlabel("The x label")
    plt.ylabel("The y label")
    pathlib.Path(f'plots').mkdir(exist_ok=True)
    pathlib.Path(f'plots/datasets/linear').mkdir(exist_ok=True)
    plt.savefig(f"plots/datasets/linear/{dataset.name}.png")
    plt.close()
    
    
@main.command()
@click.option('--dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
def plot_dataset_sin(dataset):
    plot_dataset_sin_c(dataset)
    
    
def plot_dataset_sin_c(dataset):
    dataset = pathlib.Path(dataset)
    X = np.load(dataset / "X.npy")
    y = np.load(dataset / "y.npy")
    w = np.load(dataset / "w.npy")
    x1 = X[:, 0]
    x1_ordered = sorted(x1)
    plt.plot(x1_ordered, np.sin(x1_ordered*w), '--', label=f'y=sin(pi*x)', c='y')
    plt.scatter(x1, y, label='Sampled points')
    plt.legend()
    plt.ylim(-3, 3)
    plt.title("Plot of (noisy) data points")
    plt.xlabel("The x label")
    plt.ylabel("The y label")
    pathlib.Path(f'plots').mkdir(exist_ok=True)
    pathlib.Path(f'plots/datasets/sin').mkdir(exist_ok=True)
    plt.savefig(f"plots/datasets/sin/{dataset.name}.png")
    plt.close()
    
    
@main.command()
def run():
    plot_dataset_c(dataset='datasets/linear/n250_d02_e01_seed2000')
    # plot_dataset_c(dataset='datasets/linear/n250_d02_e01_seed2000')
    # plot_dataset_c(dataset='datasets/linear/n250_d02_e01_seed3000')
    # plot_dataset_c(dataset='datasets/linear/n250_d05_e01_seed1001')
    # plot_dataset_c(dataset='datasets/linear/n250_d05_e01_seed2002')
    # plot_dataset_c(dataset='datasets/linear/n250_d05_e01_seed3003')
    # plot_dataset_c(dataset='datasets/linear/n250_d10_e01_seed1004')
    # plot_dataset_c(dataset='datasets/linear/n250_d10_e01_seed3006')
    # plot_dataset_c(dataset='datasets/linear/n250_d02_e05_seed1010')
    # plot_dataset_c(dataset='datasets/linear/n250_d02_e05_seed2010')
    # plot_dataset_c(dataset='datasets/linear/n250_d02_e05_seed3010')
    # plot_dataset_c(dataset='datasets/linear/n250_d05_e05_seed1011')
    # plot_dataset_c(dataset='datasets/linear/n250_d05_e05_seed2012')
    # plot_dataset_c(dataset='datasets/linear/n250_d05_e05_seed3013')
    # plot_dataset_c(dataset='datasets/linear/n250_d10_e05_seed1014')
    # plot_dataset_c(dataset='datasets/linear/n250_d10_e05_seed2015')
    # plot_dataset_c(dataset='datasets/linear/n250_d10_e05_seed3016')
    #plot_dataset_sin_c(dataset='datasets/sin/n250_e01_seed1000')
    #plot_dataset_sin_c(dataset='datasets/sin/n250_e05_seed2000')


if __name__ == '__main__':
    main()