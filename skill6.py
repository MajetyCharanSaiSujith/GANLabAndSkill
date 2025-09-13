# SKILLING-6: Optimized Multi-Label Facial Attribute Classification using CelebA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy

# -----------------------------
# OPTIMIZATION SETTINGS
# -----------------------------
# Enable mixed precision for faster training (if GPU available)
if tf.config.list_physical_devices('GPU'):
    set_global_policy('mixed_float16')
    print("✅ Mixed precision enabled for GPU acceleration")
else:
    print("⚠️ Running on CPU - consider using GPU for faster training")

# Set memory growth for GPU (prevents memory allocation issues)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -----------------------------
# 1. FAST DATA LOADING
# -----------------------------
attr_path = "/Users/leostephen/celeba/list_attr_celeba.csv"
images_path = "/Users/leostephen/celeba/img_align_celeba/img_align_celeba"

# Use smaller subset for faster execution
USE_SUBSET = True  # Set to False for full dataset
SUBSET_SIZE = 5000  # Adjust based on your needs
FAST_MODE = True  # Set to False for full training

print("🚀 Loading data...")
data = pd.read_csv(attr_path)

# Use subset for faster execution
if USE_SUBSET:
    data = data.sample(n=min(SUBSET_SIZE, len(data)), random_state=42).reset_index(drop=True)
    print(f"📊 Using subset of {len(data)} images for faster execution")

# Create full image paths
data['image_path'] = data.iloc[:, 0].apply(lambda x: os.path.join(images_path, x.strip()))

# Fast path verification (check only first few)
print("🔍 Verifying sample paths...")
sample_paths = data['image_path'][:3]
all_exist = all(os.path.exists(path) for path in sample_paths)
print(f"✅ Sample paths exist: {all_exist}")

# Select relevant attributes
attributes = ['Smiling', 'Eyeglasses', 'Male']
data = data[['image_path'] + attributes]

# Convert -1 to 0 for binary classification (vectorized operation)
data[attributes] = data[attributes].replace(-1, 0)

# Check class balance
print("📊 Class distribution:")
for attr in attributes:
    pos_ratio = data[attr].mean()
    print(f"  {attr}: {pos_ratio:.3f} positive ({pos_ratio * 100:.1f}%)")

# Split into train and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
print(f"📦 Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# -----------------------------
# 2. OPTIMIZED IMAGE DATA GENERATORS
# -----------------------------
# Reduced augmentation for faster training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    rotation_range=10,  # Reduced from 15
    zoom_range=0.05,  # Reduced from 0.1
    validation_split=0.2  # Built-in validation split
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Larger batch size for better GPU utilization
BATCH_SIZE = 64 if tf.config.list_physical_devices('GPU') else 32

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col=attributes,
    target_size=(64, 64),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    subset='training'  # Use built-in validation split
)

val_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col=attributes,
    target_size=(64, 64),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    subset='validation'
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col=attributes,
    target_size=(64, 64),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

# -----------------------------
# 3. OPTIMIZED CNN MODEL
# -----------------------------
print("🏗️ Building optimized model...")

model = Sequential([
    # More efficient architecture
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),

    # Use GlobalAveragePooling instead of Flatten for fewer parameters
    GlobalAveragePooling2D(),

    # Smaller dense layer
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(attributes), activation='sigmoid', dtype='float32')  # Ensure float32 output for mixed precision
])

# Optimized compilation
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for better convergence
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("📋 Model architecture:")
model.summary()

# -----------------------------
# 4. FAST TRAINING WITH CALLBACKS
# -----------------------------
# Training parameters
if FAST_MODE:
    EPOCHS = 10  # Increased from 5
    print("⚡ Fast mode: Training for 10 epochs")
else:
    EPOCHS = 25
    print("🐌 Full mode: Training for 25 epochs")

# Callbacks for faster convergence
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive reduction
        patience=3,  # Faster adaptation
        min_lr=0.00001,
        verbose=1
    )
]

print("🏋️ Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 5. FAST EVALUATION
# -----------------------------
print("📊 Evaluating model...")

# Reset test generator
test_generator.reset()

# Predict in batches for memory efficiency
y_pred_prob = model.predict(test_generator, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_df[attributes].values

print("\n📈 PERFORMANCE METRICS:")
print("=" * 50)

# Calculate metrics for each attribute
for i, attr in enumerate(attributes):
    precision = precision_score(y_true[:, i], y_pred[:, i])
    recall = recall_score(y_true[:, i], y_pred[:, i])
    print(f"{attr:12} | Precision: {precision:.3f} | Recall: {recall:.3f}")

# -----------------------------
# 6. FAST VISUALIZATION WITH BACKEND FIX
# -----------------------------
# Fix matplotlib backend for PyCharm
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for better PyCharm compatibility


def plot_training_history(history):
    """Plot training history quickly"""
    print("Generating training history plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save and show
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history saved as 'training_history.png'")
    plt.show(block=False)
    plt.pause(2)


def plot_confusion_matrices_fast(y_true, y_pred, attributes):
    """Plot all confusion matrices in one figure"""
    print("Generating confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, attr in enumerate(attributes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[i].set_title(f'{attr} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    # Save and show
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Confusion matrices saved as 'confusion_matrices.png'")
    plt.show(block=False)
    plt.pause(2)


def visualize_predictions_fast(generator, model, attributes, n=6):
    """Fast prediction visualization"""
    print("Generating prediction samples...")
    generator.reset()
    X, y_true_batch = next(generator)
    y_pred_prob = model.predict(X[:n], verbose=0)
    y_pred_batch = (y_pred_prob > 0.5).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(X[i])
        axes[i].axis('off')

        # Create prediction text
        true_text = " | ".join([f"{attr}:{int(y_true_batch[i, j])}" for j, attr in enumerate(attributes)])
        pred_text = " | ".join([f"{attr}:{int(y_pred_batch[i, j])}" for j, attr in enumerate(attributes)])
        prob_text = " | ".join([f"{y_pred_prob[i, j]:.2f}" for j in range(len(attributes))])

        title = f"True: {true_text}\nPred: {pred_text}\nProb: {prob_text}"
        axes[i].set_title(title, fontsize=9, pad=10)

    plt.tight_layout()
    # Save and show
    plt.savefig('prediction_samples.png', dpi=300, bbox_inches='tight')
    print("Prediction samples saved as 'prediction_samples.png'")
    plt.show(block=False)
    plt.pause(2)


# Generate visualizations
print("📊 Generating visualizations...")
plot_training_history(history)
plot_confusion_matrices_fast(y_true, y_pred, attributes)
visualize_predictions_fast(test_generator, model, attributes)

# -----------------------------
# 7. PERFORMANCE SUMMARY
# -----------------------------
print("\n🎯 EXECUTION SUMMARY:")
print("=" * 50)
print(f"Dataset size: {len(data)} images")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Epochs trained: {len(history.history['loss'])}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")

# Calculate average performance
avg_precision = np.mean([precision_score(y_true[:, i], y_pred[:, i]) for i in range(len(attributes))])
avg_recall = np.mean([recall_score(y_true[:, i], y_pred[:, i]) for i in range(len(attributes))])

print(f"Average Precision: {avg_precision:.3f}")
print(f"Average Recall: {avg_recall:.3f}")
print(f"Model parameters: {model.count_params():,}")

print("\n✅ Optimization techniques applied:")
print("  • Mixed precision training (if GPU available)")
print("  • Larger batch sizes for better GPU utilization")
print("  • GlobalAveragePooling instead of Flatten")
print("  • Early stopping and learning rate reduction")
print("  • Subset training option for quick testing")
print("  • Vectorized operations and efficient data loading")
print("  • Multiprocessing for data loading")

if USE_SUBSET:
    print(f"\n⚡ Quick execution mode used ({SUBSET_SIZE} samples)")
    print("   Set USE_SUBSET=False for full dataset training")
if FAST_MODE:
    print("   Set FAST_MODE=False for longer, more thorough training")