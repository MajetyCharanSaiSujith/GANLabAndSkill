import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- 1. Configuration and Setup ---
# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
IMG_SIZE = 28  # MNIST images are 28x28
BATCH_SIZE = 64
EPOCHS = 10  # MNIST trains faster, so 10 epochs is often enough
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # Digits 0-9

# --- 2. Data Loading and Transformations ---
# Define transformation pipeline
# ToTensor() converts image to tensor
# Normalize() scales tensor values. The values (0.1307,) and (0.3081,)
# are the standard mean and std deviation for the MNIST dataset.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset from torchvision
# download=True will download it if not found in the root directory.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- 3. CNN Model Definition (Adapted for MNIST) ---
class MNIST_Net(nn.Module):
    def init(self, num_classes=10):
        super(MNIST_Net, self).init()
        # Convolutional Block 1
        # in_channels=1 because MNIST is grayscale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        # The input features are 64 channels * 7 * 7 from the final pooled feature map
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize model, loss function, and optimizer
model = MNIST_Net(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Training Loop ---
print("\nStarting training on MNIST dataset...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {train_loss:.4f}")

print("Finished Training.")

# --- 5. Evaluation on Test Set ---
print("\nEvaluating on test data...")
model.eval()
correct_test = 0
total_test = 0