# ---------- Imports ----------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# ---------- Hyperparameters ----------
z_dim = 100
batch_size = 128
lr = 0.0002
epochs = 50
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Data ----------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize to [-1, 1]
])

# For real faces, replace MNIST with CelebA or LFW
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------- Generator ----------
class DCGAN_Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),   # (z_dim) -> (512,4,4)
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # -> (256,8,8)
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # -> (128,16,16)
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),     # -> (64,32,32)
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),       # -> (1,64,64)
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(-1, z_dim, 1, 1))

# ---------- Discriminator ----------
class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),   # -> (64,32,32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # -> (128,16,16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),# -> (256,8,8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),# -> (512,4,4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # -> (1,1,1)
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).view(-1, 1)  # (batch,1)

# ---------- Initialize ----------
G = DCGAN_Generator(z_dim).to(device)
D = DCGAN_Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ---------- Training ----------
for epoch in range(epochs):
    for real, _ in loader:
        real = real.to(device)
        b_size = real.size(0)
        real_label = torch.ones(b_size, 1, device=device)
        fake_label = torch.zeros(b_size, 1, device=device)

        # Train Discriminator
        z = torch.randn(b_size, z_dim, 1, 1, device=device)
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

    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # Show generated images every 10 epochs
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            sample = G(torch.randn(16, z_dim, 1, 1, device=device)).detach().cpu()
            grid = vutils.make_grid(sample, nrow=4, normalize=True, pad_value=1)
            plt.figure(figsize=(6,6))
            plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
            plt.axis("off")
            plt.show()
