import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


def load_and_prepare_data(data_root="./data", seed=42, test_size=0.2):
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=False)
    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, download=False)

    # Convert to numpy
    X_train_raw = train_dataset.data.numpy()
    y_train_raw = train_dataset.targets.numpy()
    X_test_raw = test_dataset.data.numpy()
    y_test_raw = test_dataset.targets.numpy()

    # Normalize pixel values to [0, 1]
    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0

    # Flatten images to vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Split train into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_raw, test_size=test_size, random_state=seed, stratify=y_train_raw
    )

    return {
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "X_test_raw": X_test_raw,
        "y_test_raw": y_test_raw,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test_raw,
    }


def plot_dataset_visuals(X_train_raw, y_train_raw, X_test_raw, y_test_raw, out_dir=None):
    # 1) one sample per class
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()
    for digit in range(10):
        idx = np.where(y_train_raw == digit)[0][0]
        axes[digit].imshow(X_train_raw[idx], cmap="gray")
        axes[digit].set_title(str(digit))
        axes[digit].axis("off")
    plt.tight_layout()
    if out_dir:
        plt.savefig(f"{out_dir}/samples_per_class.png", dpi=300)
    plt.close()

    # 2) class distribution
    train_counts = Counter(y_train_raw)
    classes = list(range(10))
    train_freq = [train_counts[i] for i in classes]

    plt.figure(figsize=(8, 4))
    plt.bar(classes, train_freq)
    plt.xticks(classes)
    plt.xlabel("Digit Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    if out_dir:
        plt.savefig(f"{out_dir}/class_distribution.png", dpi=300)
    plt.close()

    # 3) average image per class
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()
    for digit in range(10):
        class_images = X_train_raw[y_train_raw == digit]
        mean_image = class_images.mean(axis=0)
        axes[digit].imshow(mean_image, cmap="gray")
        axes[digit].set_title(f"Mean {digit}")
        axes[digit].axis("off")
    plt.tight_layout()
    if out_dir:
        plt.savefig(f"{out_dir}/mean_images.png", dpi=300)
    plt.close()

    # 4) pixel intensity distribution
    plt.figure(figsize=(8, 4))
    plt.hist(X_train_raw.flatten(), bins=50)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if out_dir:
        plt.savefig(f"{out_dir}/pixel_hist.png", dpi=300)
    plt.close()


def plot_tsne(X_train, y_train, out_dir=None, seed=42, subset_size=3000):
    # Randomly sample subset for t-SNE
    rng = np.random.RandomState(seed)
    subset_idx = rng.choice(len(X_train), size=subset_size, replace=False)
    X_subset = X_train[subset_idx]
    y_subset = y_train[subset_idx]

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="random",
        learning_rate="auto",
        perplexity=30
    )
    X_2d = tsne.fit_transform(X_subset)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y_subset, cmap="tab10", s=12, alpha=0.8
    )
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    if out_dir:
        plt.savefig(f"{out_dir}/tsne.png", dpi=300)
    plt.close()
