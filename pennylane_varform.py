import pennylane as qml


def rx_embedding(x, wires):
    qml.AngleEmbedding(x, wires=wires, rotation='X')


def ry_embedding(x, wires):
    qml.AngleEmbedding(x, wires=wires, rotation='Y')


def rz_embedding(x, wires):
    qml.Hadamard(wires=wires)
    qml.AngleEmbedding(x, wires=wires, rotation='Z')


def hardware_efficient_ansatz(theta, wires):
    N = len(wires)
    assert len(theta) == 2 * N
    for i in range(N):
        qml.RX(theta[2 * i], wires=wires[i])
        qml.RY(theta[2 * i + 1], wires=wires[i])
    for i in range(N-1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def tfim_ansatz(theta, wires):
    """
    Figure 6a (left) in https://arxiv.org/pdf/2105.14377.pdf
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
    tfim_ansatz(theta, wires)
    for i in range(N):
        qml.RZ(theta[2], wires=wires[i])


def zz_rx_ansatz(theta, wires):
    """
    Figure 7a in https://arxiv.org/pdf/2109.11676.pdf
    """
    N = len(wires)
    assert len(theta) == 2
    for i in range(N // 2):
        qml.MultiRZ(theta[0], wires=[wires[2 * i], wires[2 * i + 1]])
    for i in range(N // 2 - 1):
        qml.MultiRZ(theta[0], wires=[wires[2 * i + 1], wires[2 * i + 2]])
    for i in range(N):
        qml.RX(theta[1], wires=wires[i])
