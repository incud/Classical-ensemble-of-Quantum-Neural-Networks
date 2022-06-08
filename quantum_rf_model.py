########## Classical Models ##########

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.optimize import GradientDescentOptimizer
import numpy as np
import jax
import jax.numpy as jnp
import optax
rng = np.random.default_rng(12345)


class QuantumRFModel():
    def __init__(self, 
                 n_qubits, 
                 n_feature_map_layers=1, 
                 n_ansatz_layers=1, 
                 n_vrotations=3,
                 backend="lightning.qubit"):
        self.n_qubits = n_qubits
        self.n_ansatz_layers = self.n_ansatz_layers
        self.n_feature_map_layers = n_feature_map_layers
        self.n_vrotations = n_vrotations
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"
        self.device = qml.device("default.qubit.jax", wires=self.n_qubits)

    def quantumcirc(self, inputs, weights):
        @jax.jit
        @qml.qnode(self.device, interface='jax')
        def qnode(inputs, weights):
            
            #Put qubits into superposition
            for i in self.n_qubits:
                qml.Hadamard(wires=i)
                
            #Quantum ZZFeature Map
            for n in self.n_feature_map_layers:
                for i in self.n_qubits:
                    qml.RZ(2*inputs[i], wires=i)
                for i in (self.n_qubits-1):
                    qml.CNOT(wires=[i,i+1])
                    qml.RZ(2*((np.pi-inputs[i])*(np.pi-inputs[i+1])), wires=i+1)
                    qml.CNOT(wires=[i,i+1])
    
            #Ansatz
            cont = 0
            for i in self.n_qubits:
                qml.RY(weights[cont], wires=i)
                qml.RX(weights[cont+1], wires=i)
                cont = cont+2
            for n in self.n_ansatz_layers:
                for i in self.n_qubits:
                    if ((i+1)<=self.n_qubits):
                        qml.CZ(wires=[i,i+1])
                        i=i+1
                for i in self.n_qubits:
                    if (i!=0) and (i!=self.n_qubits-1):
                        qml.RY(weights[cont], wires=i)
                        qml.RX(weights[cont+1], wires=i)
                        cont = cont+2
                for i in self.n_qubits:
                    if ((i+2)<=self.n_qubits):
                        qml.CZ(wires=[i+1,i+2])
                        i=i+1
                for i in self.n_qubits:
                    qml.RY(weights[cont], wires=i)
                    qml.RX(weights[cont+1], wires=i)
                    cont = cont+2
        
            return qml.expval(qml.PauliZ(wires=0)) #measure the first qubit only, as reported in...
    
    def quantumcirc2(self, param):
        @jax.jit
        @qml.qnode(self.device, interface="jax")
        def circuit(param):
            qml.RX(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
    
    def calculate_bce_cost(self, X, Y, qnn, params, N): 
        """
        Calculate Binary Cross-Entropy
        :param X: vector of data points
        :param Y: vector of labels
        :param qnn: quantum neural network function :param params: actual parameters of the QNN model :param N: number of elements in X and Y
        :return: the BCE cost
        """
        the_cost = 0.0
        epsilon = 1e-6
        for i in range(N):
            x, y = X[i], Y[i]
            y=(y+1)/2+epsilon #1label->1;-label->0
            yp = (qnn(x, params) + 1)/2 + epsilon # 1 label -> 1; - label -> 0 
            the_cost += y * jnp.log2(yp) + (1 - y) * jnp.log2(1 - yp)
        return the_cost * (-1/N)
    '''
    def fit(self, x, y):
                
        optimizer = optax.adam(learning_rate=0.1)
        params = jax.random.normal(rng, shape=(1,1)) 
        opt_state = optimizer.init(params)
        epochs = 50
        qnn = quantumcirc2(params)
        for epoch in range(1, epochs+1):
            cost, grad_circuit = jax.value_and_grad(lambda w: calculate_mse_cost(x, y, qnn, w, N))(params)
            updates, opt_state = optimizer.update(grad_circuit, opt_state) 
            params = optax.apply_updates(params, updates)
            print(f"step {i}, cost {cost}")
            #if epoch % 50 == 0:
            #    print(".", end="", flush=True)
    
    def predict(self, x):
        return None
    '''