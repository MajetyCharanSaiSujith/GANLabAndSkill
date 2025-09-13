import os, math, time, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.datasets import fashion_mnist

# ---------- Utilities ----------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def save_image_grid(images, path, nrow=8):
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0, 1)
    N, H, W, C = images.shape
    ncol = nrow; nrow_grid = int(math.ceil(N / ncol))
    canvas = np.ones((nrow_grid * H, ncol * W, C), dtype=np.float32)
    for idx in range(N):
        r, c = idx // ncol, idx % ncol
        canvas[r*H:(r+1)*H, c*W:(c+1)*W, :] = images[idx]
    plt.figure()
    if C == 1:
        plt.imshow(canvas[...,0], cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(canvas, vmin=0, vmax=1)
    plt.axis("off"); plt.tight_layout(); plt.savefig(str(path), dpi=150); plt.close()

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ---------- Inception Score (IS) ----------
def _get_inception_model():
    try:
        model = tf.keras.applications.InceptionV3(include_top=True, weights="imagenet")
    except Exception as e:
        print("[WARN] Could not load ImageNet weights for InceptionV3. Falling back to random weights.")
        model = tf.keras.applications.InceptionV3(include_top=True, weights=None)
    return model

def _preprocess_for_inception(imgs):
    """
    imgs: float32 in [-1,1], shape (N, H, W, C)
    returns: preprocessed for InceptionV3, shape (N, 299, 299, 3), dtype float32
    """
    x = (imgs + 1.0) / 2.0  # -> [0,1], numpy
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)  # (N,H,W,3)
    x = tf.image.resize(x, (299, 299), method="bilinear").numpy()
    x = (x * 255.0).astype(np.float32)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    x = x.astype(np.float32, copy=False)
    return x

def compute_inception_score(images, splits=10, batch_size=128):
    """
    images: float32 in [-1,1], shape (N, H, W, C)
    returns: (IS_mean, IS_std)
    """
    model = _get_inception_model()
    x = _preprocess_for_inception(images)

    preds = []
    for i in range(0, len(x), batch_size):
        preds.append(model.predict(x[i:i+batch_size], verbose=0))
    p_yx = np.concatenate(preds, axis=0)
    p_yx = tf.nn.softmax(p_yx, axis=1).numpy()

    N = p_yx.shape[0]
    splits = max(1, min(int(splits), N))
    base = N // splits
    remainder = N % splits
    split_sizes = [base + (1 if i < remainder else 0) for i in range(splits)]

    scores = []
    start = 0
    for sz in split_sizes:
        part = p_yx[start:start+sz]
        start += sz
        p_y = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        kl = np.sum(kl, axis=1)
        score = np.exp(np.mean(kl))
        scores.append(score)
    return float(np.mean(scores)), float(np.std(scores))

# ---------- Models ----------
IMG_H,IMG_W,IMG_C=28,28,1; IMG_SIZE=IMG_H*IMG_W*IMG_C

def build_generator(z_dim=100,hidden=256):
    z_in=layers.Input(shape=(z_dim,), name="z")
    x=layers.Dense(hidden)(z_in); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Dense(hidden*2)(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Dense(hidden*4)(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Dense(IMG_SIZE,activation="tanh")(x); img_out=layers.Reshape((IMG_H,IMG_W,IMG_C))(x)
    return Model(inputs=z_in, outputs=img_out, name="Generator")

def build_discriminator(hidden=256):
    x_in=layers.Input(shape=(IMG_H,IMG_W,IMG_C), name="img")
    x=layers.Flatten()(x_in)
    x=layers.Dense(hidden*4)(x); x=layers.LeakyReLU(0.2)(x)
    x=layers.Dense(hidden*2)(x); x=layers.LeakyReLU(0.2)(x)
    x=layers.Dense(hidden)(x); x=layers.LeakyReLU(0.2)(x)
    logits=layers.Dense(1)(x)
    return Model(inputs=x_in, outputs=logits, name="Discriminator")

# ---------- Config ----------
class CFG: pass
CFG.out_dir=Path("data/outputs_keras_vanilla_gan_is")
CFG.epochs=20; CFG.batch_size=128; CFG.z_dim=100; CFG.hidden=256
CFG.lr=2e-4; CFG.label_smooth=0.0; CFG.sample_n=64; CFG.eval_every=5
CFG.is_samples=5000; CFG.is_splits=10
CFG.seed=42; CFG.cpu=False; CFG.limit_train=None; CFG.limit_test=None

set_seed(CFG.seed)
device="/GPU:0" if tf.config.list_physical_devices("GPU") and not CFG.cpu else "/CPU:0"
IMG_DIR,CKPT_DIR=CFG.out_dir/"images",CFG.out_dir/"checkpoints"; ensure_dir(IMG_DIR); ensure_dir(CKPT_DIR)

# ---------- Data ----------
(x_train,_),(x_test,_)=fashion_mnist.load_data()
x_train=x_train.astype("float32")/127.5-1; x_test=x_test.astype("float32")/127.5-1
x_train=np.expand_dims(x_train,-1); x_test=np.expand_dims(x_test,-1)
if CFG.limit_train: x_train=x_train[:CFG.limit_train]
if CFG.limit_test:  x_test=x_test[:CFG.limit_test]
train_ds=tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(CFG.batch_size,drop_remainder=True)

# ---------- BCE loss (with logits) ----------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# ---------- Training steps ----------
@tf.function
def d_step(D,G,batch,z_dim,d_opt,label_smooth):
    bs=tf.shape(batch)[0]
    with tf.GradientTape() as tape:
        log_r=D(batch,training=True)
        y_r=tf.ones_like(log_r)*(1.0-label_smooth)
        loss_r=bce(y_r, log_r)

        z=tf.random.normal((bs,z_dim))
        fake=G(z,training=True)
        log_f=D(fake,training=True)
        y_f=tf.zeros_like(log_f)
        loss_f=bce(y_f, log_f)

        loss=loss_r+loss_f
    grads=tape.gradient(loss,D.trainable_variables)
    d_opt.apply_gradients(zip(grads,D.trainable_variables))
    return loss

@tf.function
def g_step(D,G,bs,z_dim,g_opt):
    with tf.GradientTape() as tape:
        z=tf.random.normal((bs,z_dim))
        fake=G(z,training=True)
        log_f=D(fake,training=True)
        y_g=tf.ones_like(log_f)
        loss=bce(y_g, log_f)
    grads=tape.gradient(loss,G.trainable_variables)
    g_opt.apply_gradients(zip(grads,G.trainable_variables))
    return loss

# ---------- Train ----------
with tf.device(device):
    G,D=build_generator(CFG.z_dim,CFG.hidden),build_discriminator(CFG.hidden)
    g_opt=optimizers.Adam(CFG.lr,0.5,0.999); d_opt=optimizers.Adam(CFG.lr,0.5,0.999)
    fixed_z=tf.random.normal((CFG.sample_n,CFG.z_dim))
    hist={"g_loss":[],"d_loss":[],"is_mean":[],"is_std":[]}

    for epoch in range(1,CFG.epochs+1):
        epg=epd=cnt=0; t0=time.time()
        for batch in train_ds:
            d_loss = d_step(D,G,batch,CFG.z_dim,d_opt,CFG.label_smooth)
            g_loss = g_step(D,G,tf.shape(batch)[0],CFG.z_dim,g_opt)
            epd += float(d_loss) * batch.shape[0]
            epg += float(g_loss) * batch.shape[0]
            cnt += batch.shape[0]

        epg /= cnt; epd /= cnt
        hist["g_loss"].append(epg); hist["d_loss"].append(epd)

        # save sample grid each epoch
        samples = G(fixed_z, training=False).numpy()
        save_image_grid(samples, IMG_DIR / f"samples_epoch_{epoch:03d}.png",
                        nrow=int(CFG.sample_n**0.5))

        # Evaluate Inception Score every CFG.eval_every epochs (and on final epoch)
        if CFG.eval_every > 0 and (epoch % CFG.eval_every == 0 or epoch == CFG.epochs):
            need = CFG.is_samples
            all_fake = []; remaining = need
            while remaining > 0:
                bs = min(CFG.batch_size, remaining)
                all_fake.append(G(tf.random.normal((bs,CFG.z_dim)), training=False).numpy())
                remaining -= bs
            fake = np.concatenate(all_fake, axis=0)[:need]

            is_mean, is_std = compute_inception_score(fake, splits=CFG.is_splits, batch_size=128)
            hist["is_mean"].append(is_mean); hist["is_std"].append(is_std)
            print(f"[{epoch}] G={epg:.4f} D={epd:.4f} IS={is_mean:.3f}±{is_std:.3f} ({time.time()-t0:.1f}s)")
        else:
            print(f"[{epoch}] G={epg:.4f} D={epd:.4f} ({time.time()-t0:.1f}s)")

    save_json(hist, CFG.out_dir / "history.json")

# ---------- Plot losses ----------
plt.figure(); plt.plot(hist["g_loss"],label="G"); plt.plot(hist["d_loss"],label="D")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("GAN Losses (BCE)"); plt.show()
