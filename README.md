# Federated Learning with PySyft: A Simple Example

Federated Learning is an approach that allows multiple parties to collaborate in building a machine learning model without sharing their private data. PySyft, a library built on PyTorch, enables the implementation of Federated Learning protocols. In this blog post, we'll walk through a basic example demonstrating Federated Learning using PySyft.

## Introduction

Federated Learning facilitates collaborative model training across decentralized devices or servers while keeping data local and private. In this example, we'll create a simple linear model and train it in a federated manner using PySyft, where two virtual workers (Bob and Alice) hold separate datasets and collaborate to improve the global model without sharing their data.

## Setup and Environment

To begin, ensure you have PyTorch and PySyft installed. The code will rely on these libraries to implement federated learning. You can install them via pip:

```bash
pip install torch torchvision syft
```
## Code Implementation
The code snippet below demonstrates a basic federated learning setup using PySyft:

```bash
import torch
import syft as sy

# Hook PyTorch to PySyft
hook = sy.TorchHook(torch)

# Create virtual workers (simulating different devices)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# Create a model
model = torch.nn.Linear(2, 1)

# Send a copy of the model to the workers
model_bob = model.copy().send(bob)
model_alice = model.copy().send(alice)

# Simulated data on each worker
data_bob = torch.tensor([[1.0, 1.0], [0, 1.0]])
data_alice = torch.tensor([[1.0, 0], [0, 0]])

# Train models on each worker
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

for _ in range(10):
    # Train Bob's model
    optimizer.zero_grad()
    output_bob = model_bob(data_bob)
    loss_bob = ((output_bob - torch.tensor([[2.0], [0.0]])).pow(2)).sum()
    loss_bob.backward()
    optimizer.step()

    # Train Alice's model
    optimizer.zero_grad()
    output_alice = model_alice(data_alice)
    loss_alice = ((output_alice - torch.tensor([[1.0], [0.0]])).pow(2)).sum()
    loss_alice.backward()
    optimizer.step()

# Aggregate model updates (federated averaging)
with torch.no_grad():
    model.weight.set_((model_bob.weight.data + model_alice.weight.data) / 2)
    model.bias.set_((model_bob.bias.data + model_alice.bias.data) / 2)

# Retrieve the aggregated model
model.get()

```
## Explanation
Setting up Virtual Workers: We create two virtual workers, simulating different devices, named Sid and Stark, using the sy.VirtualWorker class provided by PySyft.

Model Initialization: A simple linear model with two input features and one output is initialized using PyTorch's torch.nn.Linear.

Sending Models to Workers: Copies of the model are sent to Bob and Alice using the .send() method, enabling them to perform computations locally.

Training Phase: Separate datasets are simulated for Sid and Stark. Each worker trains its local model using its respective data and performs gradient descent.

Aggregation: The code aggregates model updates using federated averaging to create a global model. This ensures collaboration without exchanging raw data.

Retrieval of Aggregated Model: The final aggregated model is retrieved using the .get() method.

##Conclusion
Federated Learning using PySyft offers a way to train models across decentralized datasets while maintaining data privacy. This example showcases a simplified version of federated learning, which can be expanded upon for more complex scenarios.

