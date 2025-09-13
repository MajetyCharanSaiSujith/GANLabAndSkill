import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------- Hyperparameters ----------
z_dim = 100
batch_size = 128
lr = 0.0002
epochs = 50

# ---------- Data ----------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------- Generator ----------
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.ReLU(True),
            nn.Linear(1024, 64*64), nn.Tanh()
        )
    def forward(self, z):
        return self.fc(z).view(-1, 1, 64, 64)

# ---------- Discriminator ----------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64*64, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x.view(-1, 64*64))

G = Generator(z_dim)
D = Discriminator()

# ---------- Loss & Optimizers ----------
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr)
opt_D = optim.Adam(D.parameters(), lr=lr)

# ---------- Training ----------
for epoch in range(epochs):
    for real, _ in loader:
        real = real.view(-1, 1, 64, 64)
        b_size = real.size(0)

        # Real and Fake labels
        real_label = torch.ones(b_size, 1)
        fake_label = torch.zeros(b_size, 1)

        # Train Discriminator
        z = torch.randn(b_size, z_dim)
        fake = G(z)
        D_real = D(real)
        D_fake = D(fake.detach())
        loss_D = criterion(D_real, real_label) + criterion(D_fake, fake_label)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        D_fake = D(fake)
        loss_G = criterion(D_fake, real_label)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    # Show progress
    print(f"Epoch [{epoch+1}/{epochs}]  Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")
import torchvision.utils as vutils  # add this at the top


if (epoch + 1) % 10 == 0:
    with torch.no_grad():
        sample = G(torch.randn(16, z_dim)).view(-1, 1, 64, 64)
        grid = vutils.make_grid(sample, nrow=4, normalize=True, pad_value=1)
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.show()
