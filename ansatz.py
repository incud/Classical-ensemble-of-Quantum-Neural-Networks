import pennylane as qml

def hardware_efficient_ansatz(theta, wires):
    N = len(wires)
    assert len(theta) == 3 * N
    for i in range(N):
        qml.RX(theta[3 * i], wires=wires[i])
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    #qml.CNOT(wires=[wires[N-1], wires[0]])
    for i in range(N):
        qml.RZ(theta[3 * i + 1], wires=wires[i])
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    #qml.CNOT(wires=[wires[N-1], wires[0]])
    for i in range(N):
        qml.RX(theta[3 * i + 2], wires=wires[i])
    for i in range(N-1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    #qml.CNOT(wires=[wires[N-1], wires[0]])


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
        
def get_ansatz(ansatz, n_qubits):
        if ansatz == 'hardware_efficient':
            return hardware_efficient_ansatz, 3 * n_qubits
        elif ansatz == 'tfim':
            return tfim_ansatz, 2
        elif ansatz == 'ltfim':
            return ltfim_ansatz, 3

