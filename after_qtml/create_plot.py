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
def plot_dataset(dataset, d):
    dataset = pathlib.Path(dataset)
    X = np.load(dataset / "X.npy")
    y = np.load(dataset / "y.npy")
    w = np.load(dataset / "w.npy")
    direction = d if d is not None else 0
    x1 = X[:, direction]
    plt.scatter(x1, y, label='Sampled points')
    plt.plot(x1, x1 * w[direction], '--', label=f'Slope in direction {direction}')
    plt.legend()
    plt.title("The title")
    plt.xlabel("The x label")
    plt.ylabel("The y label")
    pathlib.Path(f'plots').mkdir(exist_ok=True)
    pathlib.Path(f'plots/datasets').mkdir(exist_ok=True)
    plt.savefig(f"plots/datasets/{dataset.name}.png")


if __name__ == '__main__':
    main()