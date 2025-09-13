import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# For reproducibility
torch.manual_seed(42)

# ===============================
# 1. Load & Normalize the Dataset
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split into training and validation (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# ===========================
# 2. Class Name Mapping + Show Images
# ===========================
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def show_images():
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    images, labels = images[:12], labels[:12]

    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for i in range(12):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(class_names[labels[i]])
        ax.axis("off")

    plt.suptitle("Sample Images from Fashion MNIST", fontsize=16)
    plt.tight_layout()
    plt.show()

# Show sample images before training
show_images()

# ===============================
# 3. Define the Neural Network
# ===============================
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)  # Output layer
        )

    def forward(self, x):
        return self.model(x)

model = DNN()

# ===============================
# 4. Define Loss, Optimizer
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 weight decay

# ===============================
# 5. Early Stopping Setup
# ===============================
best_val_loss = np.inf
patience = 5
trigger_times = 0

train_losses = []
val_losses = []
max_epochs = 50

# ===============================
# 6. Training Loop
# ===============================
for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# ===============================
# 7. Plot Training and Validation Loss
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 8. Evaluate on Test Set
# ===============================
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"\n✅ Final Test Accuracy: {test_accuracy:.2f}%")