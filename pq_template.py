import pennylane as qml
import jax
from qiskit import IBMQ

# TUTORIAL
# https://docs.pennylane.ai/projects/qiskit/en/latest/devices/ibmq.html

ibm_device = 'ibmq_armonk'
ibm_token = '97c1c61552f307353f1cf0e468802933310a6a6467e910209eb086925a6842d5d9db3bc6bc0025c3183e6a3333ca139ae83e36df3406ade266f17ba360b87ddd'
n_qubits = 1

device = qml.device('qiskit.ibmq', wires=n_qubits, backend=ibm_device, ibmqx_token='XXX', hub='MYHUB', group='MYGROUP', project='MYPROJECT')
# or
provider = IBMQ.enable_account('XYZ') # https://qiskit.org/documentation/apidoc/ibmq_provider.html
dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', provider=provider)


@jax.jit
@qml.qnode(device, interface='jax')
def circuit(x, theta):
    qml.RY(x, wires=0)
    qml.RZ(theta, wires=0)
    return qml.expval(qml.PauliX(0))


print(circuit(1.1, 0.4))
print(circuit(0.8, 0.4))
print(circuit(0.8, 0.7))
