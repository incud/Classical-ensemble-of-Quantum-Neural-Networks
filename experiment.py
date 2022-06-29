import json

import click
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator
import jax
import jax.numpy as jnp
import optax
from pennylane_varform import rx_embedding, ry_embedding, rz_embedding, hardware_efficient_ansatz, tfim_ansatz, \
    ltfim_ansatz
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error


class PennyLaneModel(BaseEstimator):

    def __init__(self, var_form, layers, backend, ibm_device=None, ibm_token=None):
        assert var_form in ['hardware_efficient', 'tfim', 'ltfim']
        self.var_form = var_form
        self.layers = layers
        self.circuit = None
        self.initial_params = None
        self.params = None
        self.n_qubits = None
        assert backend in ['jax', 'ibmq']
        self.backend = backend
        self.ibm_device = ibm_device
        self.ibm_token = ibm_token
        self.seed = 12345

    def get_var_form(self, n_qubits):
        if self.var_form == 'hardware_efficient':
            return hardware_efficient_ansatz, 2 * n_qubits
        elif self.var_form == 'tfim':
            return tfim_ansatz, 2
        elif self.var_form == 'ltfim':
            return ltfim_ansatz, 3

    def create_circuit(self):
        if self.backend == 'jax':
            device = qml.device("default.qubit.jax", wires=self.n_qubits)
        elif self.backend == 'ibmq':
            device = qml.device('qiskit.ibmq', wires=self.n_qubits, backend=self.ibm_device, ibmqx_token=self.ibm_token)
        else:
            raise ValueError(f"Backend {self.backend} is unknown")
        var_form_fn, params_per_layer = self.get_var_form(self.n_qubits)

        @jax.jit
        @qml.qnode(device, interface='jax')
        def circuit(x, theta):
            ry_embedding(x, wires=range(self.n_qubits))
            for i in range(self.layers):
                var_form_fn(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(self.n_qubits))
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
        print("Fitting ", end="")
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
            print(".", end="")
        print()

    def predict(self, X):
        print("Predicting ...")
        assert len(X.shape) == 2 and X.shape[1] == self.n_qubits, f"X shape is {X.shape}, n qubits {self.n_qubits}"
        return np.array([self.circuit(xi, self.params) for xi in X])

    def get_thetas(self):

        def jnp_to_np(value):
            try:
                value_numpy = np.array(value)
                return value_numpy
            except:
                try:
                    value_numpy = np.array(value.primal)
                    return value_numpy
                except:
                    try:
                        value_numpy = np.array(value.primal.aval)
                        return value_numpy
                    except:
                        raise ValueError(f"Cannot convert to numpy value {value}")

        return jnp_to_np(self.params)


@click.group()
def main():
    pass


def save_full_plot_to_file(directory_experiment, name, scores):
    # label to plots
    x_ticks = []

    # create plot of the current bagging results
    N = len(scores['max_features'])
    for i in range(N):
        max_samples = scores['max_samples'][i]
        max_features = scores['max_features'][i]
        mse = scores['mse_bagging'][i]
        # print(max_samples, max_features, mse)
        plt.scatter([i] * len(mse['bagging_estimators']),
                    mse['bagging_estimators'],
                    color='orange', s=40)
        plt.scatter(i,
                    mse['bagging'],
                    color='red', s=45)
        x_ticks.append(f"bag s={max_samples}, f={max_features}")

    # create plot of the current boosting results
    mse = scores['mse_adaboost']
    plt.scatter([N] * len(mse['adaboost_estimators']),
                mse['adaboost_estimators'],
                color='blue', s=40)
    plt.scatter(N,
                mse['adaboost'],
                color='yellow', s=45)
    x_ticks.append("adaboost")

    plt.title(f'Testing accuracy of dataset {name}')
    plt.ylim((0, 0.5))
    plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-45)
    plt.tight_layout()
    plt.savefig(f"{directory_experiment}/{name}.png")
    plt.close('all')


@main.command()
@click.option('--directory-experiment', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--name', type=str, required=True)
def regenerate_plot(directory_experiment, name):
    score = json.load(open(f"{directory_experiment}/{name}.json"))
    save_full_plot_to_file(directory_experiment, name, score)


@main.command()
@click.option('--directory-experiment', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--directory-dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--varform', type=click.Choice(['hardware_efficient', 'tfim', 'ltfim']), required=True)
@click.option('--layers', type=click.IntRange(1, 1000), required=False, default=1)
@click.option('--n-estimators', type=click.IntRange(1, 100), required=False, default=20)
@click.option('--seed', type=click.IntRange(1, 1000000), required=True)
def run_jax(directory_experiment, directory_dataset, varform, layers, n_estimators, seed):
    specs = {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             'directory_experiment': directory_experiment,
             'directory_dataset': directory_dataset,
             'varform': varform,
             'layers': layers,
             'n_estimators': n_estimators,
             'seed': seed}
    json.dump(specs, open(f"{directory_experiment}/specs.json", "w"))

    base_estimator = PennyLaneModel(var_form=varform, layers=layers, backend='jax')

    for subdir_ds in Path(directory_dataset).iterdir():
        if subdir_ds.is_dir():
            print(f"Evaluating dataset {subdir_ds.name}")

            X_train = np.load(f"{subdir_ds}/train_X.npy")
            y_train = np.load(f"{subdir_ds}/train_y.npy")
            X_test = np.load(f"{subdir_ds}/test_X.npy")
            y_test = np.load(f"{subdir_ds}/test_y.npy")

            scores = {}
            scores['max_features'] = []
            scores['max_samples'] = []
            scores['mse_bagging'] = []
            scores['mse_adaboost'] = {}

            # evaluate bagging
            for max_samples in [0.2, 0.5, 1.0]:
                for max_features in [0.5, 1.0]:
                    bag_regr = BaggingRegressor(base_estimator=base_estimator,
                                                n_estimators=n_estimators,
                                                max_features=max_features,
                                                max_samples=max_samples,
                                                random_state=seed)
                    bag_regr.fit(X_train, y_train)
                    scores['max_samples'].append(max_samples)
                    scores['max_features'].append(max_features)
                    scores['mse_bagging'].append({
                        'bagging': mean_squared_error(y_test, bag_regr.predict(X_test)),
                        'bagging_estimators': []
                    })
                    for estimator, feature_list in zip(bag_regr.estimators_, bag_regr.estimators_features_):
                        mse_bag_estimator = mean_squared_error(y_test, estimator.predict(X_test[:, feature_list]))
                        scores['mse_bagging'][-1]['bagging_estimators'].append(mse_bag_estimator)

            # evaluate boosting
            ada_regr = AdaBoostRegressor(base_estimator=base_estimator,
                                         n_estimators=n_estimators,
                                         random_state=seed,
                                         loss='square')
            ada_regr.fit(X_train, y_train)

            scores['mse_adaboost']['adaboost'] = mean_squared_error(y_test, ada_regr.predict(X_test))
            scores['mse_adaboost']['adaboost_estimators'] = []

            for estimator in ada_regr.estimators_:
                mse_ada_estimator = mean_squared_error(y_test, estimator.predict(X_test))
                scores['mse_adaboost']['adaboost_estimators'].append(mse_ada_estimator)

            # save bagging properties to file
            name = subdir_ds.name
            print(scores)
            json.dump(scores, open(f"{directory_experiment}/{name}.json", "w"))

            save_full_plot_to_file(directory_experiment, name, scores)


@main.command()
@click.option('--directory-experiment', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--directory-dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--varform', type=click.Choice(['hardware_efficient', 'tfim', 'ltfim']), required=True)
@click.option('--layers', type=click.IntRange(1, 1000), required=False, default=1)
@click.option('--n-estimators', type=click.IntRange(1, 100), required=False, default=20)
@click.option('--seed', type=click.IntRange(1, 1000000), required=True)
@click.option('--backend', type=click.Choice(['jax', 'ibmq']), required=False, default='jax')
@click.option('--ibm-device', type=str, required=False)
@click.option('--ibm-token', type=str, required=False)
def run_jax_bag(directory_experiment, directory_dataset, varform, layers, n_estimators, seed, backend, ibm_device, ibm_token):
    specs = {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             'directory_experiment': directory_experiment,
             'directory_dataset': directory_dataset,
             'varform': varform,
             'layers': layers,
             'n_estimators': n_estimators,
             'seed': seed}
    json.dump(specs, open(f"{directory_experiment}/specs.json", "w"))

    if backend == 'jax':
        base_estimator = PennyLaneModel(var_form=varform, layers=layers, backend=backend)
    else:
        base_estimator = PennyLaneModel(var_form=varform, layers=layers, backend=backend, ibm_device=ibm_device, ibm_token=ibm_token)

    for subdir_ds in Path(directory_dataset).iterdir():
        if subdir_ds.is_dir():
            print(f"Evaluating dataset {subdir_ds.name}")

            X_train = np.load(f"{subdir_ds}/train_X.npy")
            y_train = np.load(f"{subdir_ds}/train_y.npy")
            X_test = np.load(f"{subdir_ds}/test_X.npy")
            y_test = np.load(f"{subdir_ds}/test_y.npy")

            # evaluate bagging
            max_samples = 0.2
            max_features = 0.5

            # create compact score
            scores = []

            bag_regr = BaggingRegressor(base_estimator=base_estimator,
                                        n_estimators=n_estimators,
                                        max_features=max_features,
                                        max_samples=max_samples,
                                        random_state=seed)
            bag_regr.fit(X_train, y_train)
            scores.append({
                'max_features': max_features,
                'max_samples': max_samples,
                'varform': varform,
                'layers': layers,
                'bagging_mse': mean_squared_error(y_test, bag_regr.predict(X_test)),
                'bagging_estimators_mse': [],
                'bagging_estimators_params': []
            })
            for estimator, feature_list in zip(bag_regr.estimators_, bag_regr.estimators_features_):
                mse_bag_estimator = mean_squared_error(y_test, estimator.predict(X_test[:, feature_list]))
                params_bag_estimator = estimator.get_thetas()
                scores[-1]['bagging_estimators_mse'].append(mse_bag_estimator)
                scores[-1]['bagging_estimators_params'].append(params_bag_estimator.tolist())

            # save bagging properties to file
            name = subdir_ds.name
            print(scores)
            json.dump(scores, open(f"{directory_experiment}/{name}.json", "w"))


if __name__ == '__main__':
    main()
