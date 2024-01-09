import torch
import syft as sy

# Hook PyTorch to PySyft
hook = sy.TorchHook(torch)

# Create virtual workers (simulating different devices)
Sid = sy.VirtualWorker(hook, id="Sid")
Stark = sy.VirtualWorker(hook, id="Stark")

# Create a model
model = torch.nn.Linear(2, 1)

# Send a copy of the model to the workers
model_Sid = model.copy().send(Sid)
model_Stark = model.copy().send(Stark)

# Simulated data on each worker
data_Sid = torch.tensor([[1.0, 1.0], [0, 1.0]])
data_Stark = torch.tensor([[1.0, 0], [0, 0]])

# Train models on each worker
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

for _ in range(10):
    # Train Bob's model
    optimizer.zero_grad()
    output_Sid = model_Sid(data_Sid)
    loss_Sid = ((output_Sid - torch.tensor([[2.0], [0.0]])).pow(2)).sum()
    loss_Sid.backward()
    optimizer.step()

    # Train Alice's model
    optimizer.zero_grad()
    output_Stark = model_Stark(data_Stark)
    loss_Stark = ((output_Stark - torch.tensor([[1.0], [0.0]])).pow(2)).sum()
    loss_Stark.backward()
    optimizer.step()

# Aggregate model updates (federated averaging)
with torch.no_grad():
    model.weight.set_((model_Sid.weight.data + model_Stark.weight.data) / 2)
    model.bias.set_((model_Sid.bias.data + model_Stark.bias.data) / 2)

# Retrieve the aggregated model
model.get()
