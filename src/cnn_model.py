import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model


if __name__ == "__main__":
    model = create_model()

    images = torch.randn(2, 3, 224, 224)

    model.train()
    outputs = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)

    probabilities = torch.sigmoid(outputs)

    print("First image probabilities:", probabilities[0].tolist())

# Dummy labels for our two images
labels = torch.tensor([
    [1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0]
], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss()

loss = criterion(outputs, labels)

print("Loss:", loss.item())

# Create an optimizer for the final classification layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)

# Save the current weights to check whether they change
weights_before = model.fc.weight.detach().clone()

# Perform one training update
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Verify that the weights changed
weight_change = (
    model.fc.weight.detach() - weights_before
).abs().sum().item()

print("Weight change:", weight_change)

# Calculate loss after updating the weights
with torch.no_grad():
    new_outputs = model(images)
    new_loss = criterion(new_outputs, labels)

print("Loss before update:", loss.item())
print("Loss after update:", new_loss.item())