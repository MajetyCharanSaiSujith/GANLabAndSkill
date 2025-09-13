# dcgan_cifar_subset.py
# DCGAN (Conv + ConvTranspose) for CIFAR-10 subset: cats(3), dogs(5), ships(8)
# Requires: tensorflow, numpy, matplotlib
# Tested with TF 2.x

import os, math, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import matplotlib.pyplot as plt

# ----------------- Config -----------------
class CFG: pass
CFG.out_dir = Path("./dcgan_cifar_subset"); CFG.out_dir.mkdir(exist_ok=True)
CFG.img_dir = CFG.out_dir / "images"; CFG.img_dir.mkdir(exist_ok=True)
CFG.ckpt_dir = CFG.out_dir / "checkpoints"; CFG.ckpt_dir.mkdir(exist_ok=True)

CFG.classes = [3, 5, 8]          # CIFAR-10 label ids: cat=3, dog=5, ship=8
CFG.img_h, CFG.img_w, CFG.img_c = 32, 32, 3
CFG.z_dim = 100
CFG.batch_size = 128
CFG.epochs = 100                 # change as needed (try 5-10 for debug; 100+ for quality)
CFG.lr = 2e-4
CFG.beta1 = 0.5
CFG.sample_n = 64
CFG.seed = 42

# ----------------- Utilities -----------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
set_seed(CFG.seed)

def save_image_grid(images, path, nrow=8):
    images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]
    N, H, W, C = images.shape
    ncol = nrow; nrow_grid = int(math.ceil(N / ncol))
    canvas = np.ones((nrow_grid * H, ncol * W, C), dtype=np.float32)
    for idx in range(N):
        r, c = idx // ncol, idx % ncol
        canvas[r*H:(r+1)*H, c*W:(c+1)*W, :] = images[idx]
    plt.figure(figsize=(ncol, nrow_grid))
    plt.axis('off')
    plt.imshow(np.clip(canvas, 0, 1))
    plt.savefig(str(path), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

# ----------------- Data (CIFAR-10 subset) -----------------
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
mask = np.isin(y_train, CFG.classes)
x_train = x_train[mask]
y_train = y_train[mask]
# normalize to [-1,1]
x_train = x_train.astype("float32") / 127.5 - 1.0

print(f"Using {len(x_train)} images from classes {CFG.classes}")

train_ds = tf.data.Dataset.from_tensor_slices(x_train)
train_ds = train_ds.shuffle(10000).batch(CFG.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# ----------------- Models -----------------
# Generator: z -> 4x4x512 -> upsample to 32x32x3
def build_generator(z_dim=100):
    z = layers.Input(shape=(z_dim,))
    x = layers.Dense(4*4*512, use_bias=False)(z)
    x = layers.Reshape((4,4,512))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    # 8x8
    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    # 16x16
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    # 32x32
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    out = layers.Conv2D(CFG.img_c, kernel_size=3, padding="same", activation="tanh")(x)  # [-1,1]
    return Model(z, out, name="Generator")

# Discriminator: 32x32x3 -> downsample -> logits
def build_discriminator():
    img = layers.Input(shape=(CFG.img_h, CFG.img_w, CFG.img_c))
    x = layers.Conv2D(64, 4, strides=2, padding="same")(img)   # 16x16
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False)(x)   # 8x8
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False)(x)   # 4x4
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)   # logits
    return Model(img, x, name="Discriminator")

# ----------------- Losses, Opts -----------------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def d_loss_fn(real_logits, fake_logits):
    real_labels = tf.ones_like(real_logits)
    fake_labels = tf.zeros_like(fake_logits)
    loss_real = bce(real_labels, real_logits)
    loss_fake = bce(fake_labels, fake_logits)
    return loss_real + loss_fake

def g_loss_fn(fake_logits):
    real_labels = tf.ones_like(fake_logits)
    return bce(real_labels, fake_logits)

g_opt = optimizers.Adam(CFG.lr, beta_1=CFG.beta1)
d_opt = optimizers.Adam(CFG.lr, beta_1=CFG.beta1)

# ----------------- Training step -----------------
@tf.function
def train_step(real_images):
    bs = tf.shape(real_images)[0]
    z = tf.random.normal((bs, CFG.z_dim))
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = G(z, training=True)
        real_logits = D(real_images, training=True)
        fake_logits = D(fake_images, training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)
        g_loss = g_loss_fn(fake_logits)

    d_grads = d_tape.gradient(d_loss, D.trainable_variables)
    g_grads = g_tape.gradient(g_loss, G.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, D.trainable_variables))
    g_opt.apply_gradients(zip(g_grads, G.trainable_variables))
    return d_loss, g_loss

# ----------------- Build models & training loop -----------------
G = build_generator(CFG.z_dim)
D = build_discriminator()
G.summary()
D.summary()

fixed_z = tf.random.normal((CFG.sample_n, CFG.z_dim))

# optional checkpointing
ckpt = tf.train.Checkpoint(generator=G, discriminator=D, g_opt=g_opt, d_opt=d_opt)
ckpt_manager = tf.train.CheckpointManager(ckpt, str(CFG.ckpt_dir), max_to_keep=3)

# training
hist = {"d_loss":[], "g_loss":[]}
for epoch in range(1, CFG.epochs + 1):
    t0 = time.time()
    d_epoch = 0.0; g_epoch = 0.0; steps = 0
    for batch in train_ds:
        d_loss, g_loss = train_step(batch)
        d_epoch += float(d_loss); g_epoch += float(g_loss); steps += 1

    d_epoch /= steps; g_epoch /= steps
    hist["d_loss"].append(d_epoch); hist["g_loss"].append(g_epoch)

    # sample & save
    samples = G(fixed_z, training=False).numpy()
    save_image_grid(samples, CFG.img_dir / f"samples_epoch_{epoch:03d}.png", nrow=int(math.sqrt(CFG.sample_n)))

    # checkpoint every 10 epochs
    if epoch % 10 == 0:
        ckpt_manager.save()

    print(f"[{epoch}/{CFG.epochs}] D_loss={d_epoch:.4f} G_loss={g_epoch:.4f} time={(time.time()-t0):.1f}s")

# final save
G.save(str(CFG.out_dir / "generator_final.h5"))
D.save(str(CFG.out_dir / "discriminator_final.h5"))

# plot losses (simple)
import matplotlib.pyplot as plt
plt.plot(hist["g_loss"], label="G_loss"); plt.plot(hist["d_loss"], label="D_loss")
plt.legend(); plt.title("Losses"); plt.xlabel("Epoch"); plt.savefig(str(CFG.out_dir/"losses.png"), dpi=150)
