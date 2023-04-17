import pennylane as qml
import jax.numpy as jnp

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
        qml.RY(jnp.arccos(x[i]), wires=i)
    for i in range(len(wires)):
        qml.RZ(jnp.arcsin(x[i]**2), wires=i)