import json
import click
import numpy as np
import matplotlib.pyplot as plt


@click.group()
def main():
    pass


@main.command()
@click.option('--dir', type=click.Path(exists=True, dir_okay=True, file_okay=False), multiple=True, required=True)
def run_experiment(dir):
    points = {}
    ens_points = {}
    for directory in dir:
        x = int(json.load(open(f"{directory}/specs.json"))['n_layers'])
        y = np.load(f"{directory}/testing_losses.npy")
        points[x] = y
        ens_points[x] = np.load(f"{directory}/testing_ensemble_avg_losses.npy")

    print(points)
    print()

    xl = list(points.keys())
    xl.sort()
    yl = [np.average(points[x]) for x in xl]
    ye = [np.std(points[x]) for x in xl]
    print(xl)
    print(yl)
    print(ye)
    plt.errorbar(xl, yl, ye, marker='s', color='blue')
    yens = [ens_points[x] for x in xl]
    plt.plot(xl, yens, color='red')
    plt.show()


if __name__ == '__main__':
    main()
