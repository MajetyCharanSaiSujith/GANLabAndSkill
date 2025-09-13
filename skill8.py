# skill8.py
import os
import random
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image

# ===========================
# 1. Download dataset via KaggleHub
# ===========================
path = kagglehub.dataset_download("splcher/animefacedataset")
print("Dataset downloaded to:", path)

# IMPORTANT: inside this folder there is an "images" subfolder
dataset_path = os.path.join(path, "images")

# ===========================
# 2. Custom Dataset (limit to 200 images)
# ===========================
class LimitedImageFolder(Dataset):
    def __init__(self, root, transform=None, limit=200):
        self.base_dataset = ImageFolder(root=root, transform=transform)
        self.limit = min(limit, len(self.base_dataset))

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        return self.base_dataset[idx]

# ===========================
# 3. Hyperparameters
# ===========================
image_size = 64
batch_size = 32
z_dim = 100
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = LimitedImageFolder(dataset_path, transform=transform, limit=200)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Loaded images:", len(dataset))

# ===========================
# 4. Generator and Discriminator
# ===========================
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*8, feature_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*4, feature_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*2, feature_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*4, feature_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.net(x).view(-1)

# ===========================
# 5. Loss Functions
# ===========================
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()

def hinge_d_loss(real_pred, fake_pred):
    return torch.mean(torch.relu(1.0 - real_pred)) + torch.mean(torch.relu(1.0 + fake_pred))

def hinge_g_loss(fake_pred):
    return -torch.mean(fake_pred)

# ===========================
# 6. Training Function
# ===========================
def train_gan(loss_type="BCE"):
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)

    print(f"\n===== Training {loss_type} GAN =====")
    for epoch in range(num_epochs):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)
            bs = real.size(0)

            # Train Discriminator
            noise = torch.randn(bs, z_dim, 1, 1, device=device)
            fake = G(noise).detach()
            real_pred = D(real)
            fake_pred = D(fake)

            if loss_type == "BCE":
                d_loss = bce_loss(real_pred, torch.ones_like(real_pred)) + \
                         bce_loss(fake_pred, torch.zeros_like(fake_pred))
            elif loss_type == "LSGAN":
                d_loss = mse_loss(real_pred, torch.ones_like(real_pred)) + \
                         mse_loss(fake_pred, torch.zeros_like(fake_pred))
            elif loss_type == "HINGE":
                d_loss = hinge_d_loss(real_pred, fake_pred)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Train Generator
            noise = torch.randn(bs, z_dim, 1, 1, device=device)
            fake = G(noise)
            fake_pred = D(fake)

            if loss_type == "BCE":
                g_loss = bce_loss(fake_pred, torch.ones_like(fake_pred))
            elif loss_type == "LSGAN":
                g_loss = mse_loss(fake_pred, torch.ones_like(fake_pred))
            elif loss_type == "HINGE":
                g_loss = hinge_g_loss(fake_pred)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            if i % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} "
                      f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

        # Save sample images
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        save_image(fake, f"samples_{loss_type}_epoch{epoch+1}.png", normalize=True, nrow=4)

# ===========================
# 7. Run Training
# ===========================
train_gan("BCE")
train_gan("LSGAN")
train_gan("HINGE")
