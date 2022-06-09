
import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
import optax

def RZZ(theta, wires):
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(theta, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def LaroccaZXForm(theta, n_layers, wires):
    """
    Create the template for variational form in 'The theory of overparameterization' Figure 7a
    :param theta:
    :param n_layers:
    :param wires:
    :return:
    """
    assert len(theta) == 2 * n_layers, "The number of theta parameters must be 2 * n_layers"
    for i in range(len(wires)):
        qml.Hadamard(wires=wires[i])
    for l in range(n_layers):
        for i in range(len(wires)//2):
            RZZ(theta[2 * l], wires=[2 * i, 2 * i + 1])
        for i in range(len(wires) // 2 - 1):
            RZZ(theta[2*l], wires=[2 * i + 1, 2 * i + 2])
        for i in range(len(wires)):
            qml.RX(theta[2*l+1], wires=i)


class QuantumExperiment:
    """
    Instantiate and train multiple times the quantum circuit, save results in a new directory.
    TODO it should extend Classifier or Regressor class of scipy
    """
    def __init__(self, directory, qasm_file=None):
        """
        Constructor
        :param X_train: training set data
        :param y_train: training set labels
        :param qasm_file: TODO
        """
        self.directory = directory
        self.circuit = None  # TODO load from qasm file
        self.num_theta = None
        self.initial_parameters_vector = []
        self.trained_parameters_vector = []
        self.training_loss_vector = []
        self.testing_loss_vector = []
        self.testing_loss_ensemble_avg = None

    def create_circuit(self, n_qubits, angle_embedding_rotation='X', variational_form='', variational_layers=1):
        """

        :param angle_embedding_rotation:
        :param variational_form:
        :param variational_layers:
        :return:
        """
        device = qml.device("default.qubit.jax", wires=n_qubits)

        @jax.jit
        @qml.qnode(device, interface='jax')
        def circuit(x, theta):
            qml.AngleEmbedding(features=x, rotation=angle_embedding_rotation, wires=range(n_qubits))
            LaroccaZXForm(theta=theta, n_layers=variational_layers, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(wires=0))

        self.circuit = circuit
        self.num_theta = 2 * variational_layers

    @classmethod
    def mse_loss(cls, predictor, theta, X, y):
        return np.sum((yi - predictor(xi, theta))**2 for (xi, yi) in zip(X, y)) / (2*len(X))

    def train_new_iid_circuit(self, X_train, y_train, epochs=200, init_state_seed=12345):
        optimizer = optax.adam(learning_rate=0.1)
        init_params = jax.random.normal(jax.random.PRNGKey(init_state_seed), shape=(self.num_theta,))
        self.initial_parameters_vector.append(init_params)

        params = init_params.copy()
        opt_state = optimizer.init(params)

        for epoch in range(epochs):
            loss, grad_circuit = jax.value_and_grad(lambda theta: self.mse_loss(self.circuit, theta, X_train, y_train))(params)
            updates, opt_state = optimizer.update(grad_circuit, opt_state)
            params = optax.apply_updates(params, updates)
            print(f"epoch {epoch+1}, cost {loss}")

        self.trained_parameters_vector.append(params)
        self.training_loss_vector.append(loss)

    def evaluate_iid_circuits(self, X_test, y_test):
        for trained_theta in self.trained_parameters_vector:
            loss, _ = jax.value_and_grad(lambda theta: self.mse_loss(self.circuit, theta, X_test, y_test))(trained_theta)
            self.testing_loss_vector.append(loss)

    def evaluate_ensemble_avg(self, X_test, y_test):
        def ensemble_avg(x):
            return np.sum(self.circuit(x, theta) for theta in self.trained_parameters_vector) / len(self.trained_parameters_vector)
        ensemble_avg_wrap = lambda x, theta: ensemble_avg(x)
        self.testing_loss_ensemble_avg = self.mse_loss(ensemble_avg_wrap, None, X_test, y_test)

    def save(self):
        # TODO save metadata to JSON
        np.save(f"{self.directory}/initial_params.npy", np.array(self.initial_parameters_vector))
        np.save(f"{self.directory}/trained_params.npy", np.array(self.trained_parameters_vector))
        np.save(f"{self.directory}/training_losses.npy", np.array(self.training_loss_vector))
        np.save(f"{self.directory}/testing_losses.npy", np.array(self.testing_loss_vector))
        np.save(f"{self.directory}/testing_ensemble_avg_losses.npy", np.array([self.testing_loss_ensemble_avg]))
