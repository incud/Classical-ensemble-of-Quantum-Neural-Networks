import click
import pennylane as qml
import numpy as np
from sklearn.base import BaseEstimator
import jax
import jax.numpy as jnp
import optax
from pennylane_varform import rx_embedding, ry_embedding, rz_embedding, hardware_efficient_ansatz, tfim_ansatz, ltfim_ansatz, zz_rx_ansatz


class PennyLaneModel(BaseEstimator):

    def __init__(self, var_form, layers, seed):
        assert var_form in ['hardware_efficient', 'tfim', 'ltfim', 'zz_rx']
        self.var_form = var_form
        self.layers = layers
        self.circuit = None
        self.initial_params = None
        self.params = None
        self.n_qubits = None
        self.seed = seed

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


if __name__ == '__main__':
    main()

