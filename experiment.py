import json

import click
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator
import jax
import jax.numpy as jnp
import optax
from pennylane_varform import rx_embedding, ry_embedding, rz_embedding, hardware_efficient_ansatz, tfim_ansatz, ltfim_ansatz, zz_rx_ansatz
from sklearn.ensemble import BaggingRegressor
from pathlib import Path


class PennyLaneModel(BaseEstimator):

    def __init__(self, var_form, layers):
        assert var_form in ['hardware_efficient', 'tfim', 'ltfim', 'zz_rx']
        self.var_form = var_form
        self.layers = layers
        self.circuit = None
        self.initial_params = None
        self.params = None
        self.n_qubits = None
        self.seed = 12345

    def get_var_form(self, n_qubits):
        if self.var_form == 'hardware_efficient':
            return hardware_efficient_ansatz, 2 * n_qubits
        elif self.var_form == 'tfim':
            return tfim_ansatz, 2
        elif self.var_form == 'ltfim':
            return ltfim_ansatz, 3
        elif self.var_form == 'zz_rx':
            return zz_rx_ansatz, 2

    def create_circuit(self):
        device = qml.device("default.qubit.jax", wires=self.n_qubits)
        var_form_fn, params_per_layer = self.get_var_form(self.n_qubits)

        @jax.jit
        @qml.qnode(device, interface='jax')
        def circuit(x, theta):
            ry_embedding(x, wires=range(self.n_qubits))
            for i in range(len(self.layers)):
                var_form_fn(theta[i * params_per_layer, (i + 1) * params_per_layer], wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(wires=0))

        self.circuit = circuit
        self.initial_params = jax.random.normal(jax.random.PRNGKey(self.seed), shape=(self.layers * params_per_layer,))

    def calculate_mse_cost(self, X, y, theta):
        n = X.shape[0]
        cost = 0
        for i in range(n):
            x_i, y_i = X[i], y[i]
            yp = self.circuit(x_i, theta)
            cost += (yp - y_i) ** 2
        return cost / (2 * n)

    def fit(self, X, y):
        assert len(X.shape) == 2 and len(y.shape) == 1 and X.shape[0] == y.shape[0]
        self.n_qubits = X.shape[1]
        self.create_circuit()
        self.params = jnp.copy(self.initial_params)
        optimizer = optax.adam(learning_rate=0.1)
        opt_state = optimizer.init(self.initial_params)
        epochs = 150
        for epoch in range(epochs):
            cost, grad_circuit = jax.value_and_grad(lambda theta: self.calculate_mse_cost(X, y, theta))(self.params)
            updates, opt_state = optimizer.update(grad_circuit, opt_state)
            self.params = optax.apply_updates(self.params, updates)

    def transform(self, X):
        assert len(X.shape) == 2 and X.shape[1] == self.n_qubits
        return [self.circuit(xi, self.params) for xi in X]


@click.group()
def main():
    pass


@main.command()
@click.option('--directory-experiment', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--directory-dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--varform', type=click.Choice(['hardware_efficient', 'tfim', 'ltfim', 'zz_rx']), required=True)
@click.option('--layers', type=click.IntRange(1, 1000), required=False, default=1)
@click.option('--n-estimators', type=click.IntRange(1, 100), required=False, default=20)
@click.option('--seed', type=click.IntRange(1, 1000000), required=False, default=12345)
def run_jax(directory_experiment, directory_dataset, varform, layers, n_estimators, seed):

    base_estimator = PennyLaneModel(var_form=varform, layers=layers)

    for subdir_ds in Path(directory_dataset).iterdir():
        if subdir_ds.is_dir():

            # evaluate bagging properties
            scores = {}
            for max_samples in [0.2, 0.5, 1.0]:
                for max_features in [0.5, 1.0]:
                    X_train = np.load(f"{subdir_ds}/train_X.npy")
                    y_train = np.load(f"{subdir_ds}/train_y.npy")
                    X_test = np.load(f"{subdir_ds}/test_X.npy")
                    y_test = np.load(f"{subdir_ds}/test_y.npy")

                    regr = BaggingRegressor(base_estimator=base_estimator,
                                            n_estimators=n_estimators,
                                            max_features=max_features,
                                            max_samples=max_samples,
                                            random_state=seed)
                    regr.fit(X_train, y_train)
                    scores[(max_samples, max_features)] = {}
                    scores[(max_samples, max_features)]['bagging'] = regr.score(X_test, y_test)
                    for i, estimator in enumerate(regr.estimators_):
                        scores[(max_samples, max_features)]['estimators'] = []
                        scores[(max_samples, max_features)]['estimators'][i] = regr.score(X_test, y_test)

            # save bagging properties to file
            name = subdir_ds.name
            json.dump(scores, open(f"experiments_jax/{directory_experiment}/{name}.json"))

            # create plot of the current bagging results
            x_ticks = []
            for i, (max_samples, max_features) in enumerate(scores.keys()):
                plt.scatter(i,
                            scores[(max_samples, max_features)]['bagging'],
                            color='red')
                plt.scatter([i] * n_estimators,
                            [scores[(max_samples, max_features)]['estimators'][j] for j in range(n_estimators)],
                            color='blue')
                x_ticks.append(f"s={max_samples}, f={max_features}")

            plt.title(f'Testing accuracy of dataset {name}')
            plt.xticks(range(len(scores.keys())), x_ticks)
            plt.savefig(f"experiments_jax/{directory_experiment}/{name}.png")
            plt.close('all')


if __name__ == '__main__':
    main()

