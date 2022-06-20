import json

import click
import sklearn.utils
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dataset_gaussian import GaussianMixtureClassificationDataset
from sklearn.model_selection import train_test_split
from quantum_experiment import QuantumExperiment



@click.group()
def main():
    pass


@main.command()
@click.option('--d', type=int, required=True)
@click.option('--n', type=int, required=True)
@click.option('--n-subsample', type=int, required=True)
@click.option('--n-ensemble', type=int, required=True)
@click.option('--n-layers', type=int, required=True)
@click.option('--epochs', type=int, required=True)
@click.option('--seed', type=int, required=True)
def run_experiment(d, n, n_subsample, n_ensemble, n_layers, epochs, seed):
    specs = {'d': d, 'n': n, 'n_subsample': n_subsample, 'n_ensemble': n_ensemble, 'n_layers': n_layers, 'epochs': epochs, 'seed': seed}


    np.random.seed(seed)
    directory = f"experiment_d{d}_nens{n_ensemble}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = Path(directory)
    path.mkdir(exist_ok=False)
    print("Creating dataset")
    gm = GaussianMixtureClassificationDataset(n, d, 0, 1.0, 0)
    print("Split into training and testing set")
    X_train, X_test, y_train, y_test = train_test_split(np.array(gm.X_noise), np.array(gm.y), test_size=0.90)
    with open(f"{directory}/specs.json", 'w') as outfile:
        json.dump(specs, outfile)
    np.save(f"{directory}/X_train.npy", X_train)
    np.save(f"{directory}/X_test.npy", X_test)
    np.save(f"{directory}/y_train.npy", y_train)
    np.save(f"{directory}/y_test.npy", y_test)
    print("Instantiation of quantum circuit")
    qe = QuantumExperiment(directory)
    n_qubits = X_train.shape[1]
    qe.create_circuit(n_qubits=n_qubits, angle_embedding_rotation='X', variational_layers=n_layers)
    print("Running training single predictors")
    for i in range(n_ensemble):
        X_train_subsampled, y_train_subsampled = sklearn.utils.resample(X_train, y_train, n_samples=n_subsample, replace=False)
        qe.train_new_iid_circuit(X_train_subsampled, y_train_subsampled, epochs=epochs, init_state_seed=np.random.randint(100000))
        print(f"Finished {i+1}-th training")
    print("Testing single predictors")
    qe.evaluate_iid_circuits(X_test, y_test)
    print("Testing ensemble")
    qe.evaluate_ensemble_avg(X_test, y_test)
    print("Saving to file")
    qe.save()


@main.command()
@click.option('--dir', type=click.Path(exists=True, dir_okay=True, file_okay=False), multiple=True, required=True)
def run_experiment(dir):
    points = {}
    for directory in dir:
        x = int(json.load(open(f"{directory}/specs.json"))['n_layers'])
        y = np.load(f"{directory}/testing_losses.npy")[-1]
        if x in points:
            points[x].append(y)
        else:
            points[x] = [y]

    print(points)
    print()
    
    xl = list(points.keys())
    xl.sort()
    yl = [np.average(points[x]) for x in xl]
    ye = [np.std(points[x]) for x in xl]
    print(xl)
    print(yl)
    print(ye)
    plt.errorbar(xl, yl, ye, marker='s')
    plt.show()


if __name__ == '__main__':
    main()
