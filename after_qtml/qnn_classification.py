import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import CrossEntropyLoss

IBM_QISKIT_HUB = 'MYHUB'
IBM_QISKIT_GROUP = 'MYGROUP'
IBM_QISKIT_PROJECT = 'MYPROJECT'


def rx_embedding(x, wires):
    qml.AngleEmbedding(x, wires=wires, rotation='X')

def ry_embedding(x, wires):
    qml.AngleEmbedding(x, wires=wires, rotation='Y')

def rz_embedding(x, wires):
    qml.Hadamard(wires=wires)
    qml.AngleEmbedding(x, wires=wires, rotation='Z')
    
def embedding_sin(x, wires):
    for i in wires:
        qml.RY(jnp.arcsin(x), wires=i)
    for i in wires:
        qml.RZ(jnp.arccos(x**2), wires=i)
    
def embedding(x, wires):
    for i in range(len(wires)):
        qml.RY(jnp.arcsin(x[i]), wires=i)
    for i in wires:
        qml.RZ(jnp.arccos(x[i]**2), wires=i)


def hardware_efficient_ansatz(theta, wires):
    N = len(wires)
    assert len(theta) == 3 * N
    for i in range(N):
        qml.RX(theta[3 * i], wires=wires[i])
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    for i in range(N):
        qml.RZ(theta[3 * i + 1], wires=wires[i])
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    for i in range(N):
        qml.RX(theta[3 * i + 2], wires=wires[i])


def tfim_ansatz(theta, wires):
    """
    Figure 6a (left) in https://arxiv.org/pdf/2105.14377.pdf
    Figure 7a in https://arxiv.org/pdf/2109.11676.pdf
    """
    N = len(wires)
    assert len(theta) == 2
    for i in range(N//2):
        qml.MultiRZ(theta[0], wires=[wires[2 * i], wires[2 * i + 1]])
    for i in range(N // 2 - 1):
        qml.MultiRZ(theta[0], wires=[wires[2 * i + 1], wires[2 * i + 2]])
    for i in range(N):
        qml.RX(theta[1], wires=wires[i])


def ltfim_ansatz(theta, wires):
    """
    Figure 6a (right) in https://arxiv.org/pdf/2105.14377.pdf
    """
    N = len(wires)
    assert len(theta) == 3
    tfim_ansatz(theta[:2], wires)
    for i in range(N):
        qml.RZ(theta[2], wires=wires[i])


class Qnn(BaseEstimator):

    def __init__(self, var_form, layers, backend, seed, ibm_device=None, ibm_token=None):
        assert var_form in ['hardware_efficient', 'tfim', 'ltfim']
        assert backend in ['jax', 'ibmq']
        self.var_form = var_form
        self.layers = layers
        self.circuit = None
        self.initial_params = None
        self.params = None
        self.n_qubits = None
        self.backend = backend
        self.ibm_device = ibm_device
        self.ibm_token = ibm_token
        self.seed = seed
        self.random_state = self.seed

    def get_var_form(self, n_qubits):
        if self.var_form == 'hardware_efficient':
            return hardware_efficient_ansatz, 3 * n_qubits
        elif self.var_form == 'tfim':
            return tfim_ansatz, 2
        elif self.var_form == 'ltfim':
            return ltfim_ansatz, 3

    def create_circuit(self):
        if self.backend == 'jax':
            device = qml.device("default.qubit.jax", wires=self.n_qubits)
        elif self.backend == 'ibmq':
            device = qml.device('qiskit.ibmq', wires=self.n_qubits, backend=self.ibm_device,
                                ibmqx_token=self.ibm_token, hub=IBM_QISKIT_HUB,
                                group=IBM_QISKIT_GROUP, project=IBM_QISKIT_PROJECT)
        else:
            raise ValueError(f"Backend {self.backend} is unknown")
        var_form_fn, params_per_layer = self.get_var_form(self.n_qubits)

        @jax.jit
        @qml.qnode(device, interface='jax')
        def circuit(x, theta):
            for i in range(self.layers):
                embedding(x, wires=range(self.n_qubits))
                var_form_fn(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(wires=0))

        self.circuit = circuit
        self.initial_params = jax.random.normal(jax.random.PRNGKey(self.random_state), shape=(self.layers * params_per_layer,))

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
        #print('init params',self.initial_params)
        optimizer = optax.adam(learning_rate=0.1)
        opt_state = optimizer.init(self.initial_params)
        epochs = 150
        for epoch in range(epochs):
            cost, grad_circuit = jax.value_and_grad(lambda theta: self.calculate_mse_cost(X, y, theta))(self.params)
            updates, opt_state = optimizer.update(grad_circuit, opt_state)
            #print(self.params)
            self.params = optax.apply_updates(self.params, updates)
            # print('params',self.params)
            # print('cost',cost)
            print(".", end="")
        print()

    def predict(self, X):
        print("Predicting ...")
        assert len(X.shape) == 2 and X.shape[1] == self.n_qubits, f"X shape is {X.shape}, n qubits {self.n_qubits}"
        #print(self.params)
        #for i in range(len(X)):
            # print()
            # print('x_i',X[i])
            # print('y_pred',self.circuit(X[i], self.params))
            # print('y_true',np.sin(X[i]),np.sin(np.pi*X[i]),y[i])
            # print()
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