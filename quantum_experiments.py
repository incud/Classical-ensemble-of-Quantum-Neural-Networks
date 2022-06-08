from dataset import GaussianMixtureClassificationDataset
from quantum_rf_model import QuantumRFModel
import random
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
rng = np.random.default_rng(12345)


random.seed(10)
np.random.seed(10)

n_train = 10
n_test = 10
n = n_train + n_test
d = 2
d_prime = 100
padding = 0
epsilon_d = 0
epsilon_padding = 0
n_qubits = d+padding
n_ansatz_layers = 1
n_feature_map_layers = 1


data = GaussianMixtureClassificationDataset(n, d, padding, epsilon_d, epsilon_padding)
x = np.array(data.X_noise)
y = np.array(data.y)

device = qml.device("default.qubit.jax", wires=n_qubits)
@jax.jit
@qml.qnode(device, interface='jax')
def circuit(inputs, weights):
    
    #Put qubits into superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        
    #Quantum ZZFeature Map
    for n in range(n_feature_map_layers):
        for i in range(n_qubits):
            qml.RZ(2*inputs[i], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i,i+1])
            qml.RZ(2*((np.pi-inputs[i])*(np.pi-inputs[i+1])), wires=i+1)
            qml.CNOT(wires=[i,i+1])

    #Ansatz
    cont = 0
    for i in range(n_qubits):
        qml.RY(weights[cont], wires=i)
        qml.RX(weights[cont+1], wires=i)
        cont = cont+2
    for n in range(n_ansatz_layers):
        for i in range(n_qubits):
            if ((i+1)<n_qubits):
                qml.CZ(wires=[i,i+1])
                i=i+1
        for i in range(n_qubits):
            if (i!=0) and (i!=n_qubits-1):
                qml.RY(weights[cont], wires=i)
                qml.RX(weights[cont+1], wires=i)
                cont = cont+2
        for i in range(n_qubits):
            if ((i+2)<n_qubits):
                qml.CZ(wires=[i+1,i+2])
                i=i+1
        for i in range(n_qubits):
            qml.RY(weights[cont], wires=i)
            qml.RX(weights[cont+1], wires=i)
            cont = cont+2

    return qml.expval(qml.PauliZ(wires=0))


def calculate_mse_cost(x,y,circuit,w,n):
    cost = 0
    for i in range(n):
        x_i, y_i = x[i], y[i]
        yp = circuit(x_i,w)
        cost += (yp-y_i) ** 2
    return cost/(2*n)

optimizer = optax.adam(learning_rate=0.1)
num_params=2*n_qubits+n_ansatz_layers*(4*n_qubits-4)
params = jax.random.normal(jax.random.PRNGKey(0),shape=(num_params,)) 
opt_state = optimizer.init(params)
epochs = 1000
for epoch in range(1, epochs+1):
    cost, grad_circuit = jax.value_and_grad(lambda w: calculate_mse_cost(x,y,circuit,w,n))(params)
    updates, opt_state = optimizer.update(grad_circuit, opt_state) 
    params = optax.apply_updates(params, updates)
    print(f"epoch {epoch}, cost {cost}")
    #if epoch % 50 == 0:
    #    print(".", end="", flush=True)
    
    
