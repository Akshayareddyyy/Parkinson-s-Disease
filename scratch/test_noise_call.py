import pennylane as qml
import torch
import time

dev = qml.device('default.mixed', wires=9)

@qml.qnode(dev, interface='torch')
def circuit(weights, x):
    for i in range(9):
        qml.RX(x[i], wires=i)
        qml.DepolarizingChannel(0.001, wires=i)
    for l in range(2):
        for i in range(9):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
            qml.DepolarizingChannel(0.001, wires=i)
        for i in range(8):
            qml.CNOT(wires=[i, i+1])
            qml.DepolarizingChannel(0.001, wires=i)
            qml.DepolarizingChannel(0.001, wires=i+1)
        qml.CNOT(wires=[8, 0])
        qml.DepolarizingChannel(0.001, wires=8)
        qml.DepolarizingChannel(0.001, wires=0)
    return [qml.expval(qml.PauliZ(i)) for i in range(9)]

weights = torch.zeros((2, 9, 2), dtype=torch.float64)
x_single = torch.zeros(9, dtype=torch.float64)

t0 = time.time()
print("Calling 1 sample...", flush=True)
res1 = circuit(weights, x_single)
print(f"1 sample finished in {time.time()-t0:.2f}s", flush=True)

x_batch = torch.zeros(9, 4, dtype=torch.float64)
t0 = time.time()
print("Calling batch of 4...", flush=True)
try:
    res_b = circuit(weights, x_batch)
    print(f"Batch of 4 finished in {time.time()-t0:.2f}s", flush=True)
except Exception as e:
    print(f"Batch failed: {e}", flush=True)

