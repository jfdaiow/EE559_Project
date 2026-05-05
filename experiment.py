import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from data_processing import load_and_prepare_data, plot_dataset_visuals, plot_tsne

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

output_dir = "mnist_results"
fig_dir = os.path.join(output_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)


def run_knn_experiment(X_train, y_train, X_val, y_val, X_test, y_test, fig_dir):
    k_values = [1, 3, 5, 7, 9, 11]
    val_accs, val_f1s = [], []

    # Try different k values
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        # Train model
        t0 = time.time()
        knn.fit(X_train, y_train)
        fit_time = time.time() - t0
        t1 = time.time()
        # Predict on validation set
        y_val_pred = knn.predict(X_val)
        pred_time = time.time() - t1

        # Compute metrics
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average="macro")
        val_accs.append(acc)
        val_f1s.append(f1)

        print(f"k={k:2d} | val acc={acc:.4f} | val F1={f1:.4f} | fit={fit_time:.2f}s | pred={pred_time:.2f}s")

    # Select best k
    best_k = k_values[int(np.argmax(val_accs))]
    print("Best k =", best_k)

    plt.figure(figsize=(6, 4))
    plt.plot(k_values, val_accs, marker="o", label="Validation Accuracy")
    plt.plot(k_values, val_f1s, marker="s", label="Validation F1")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title("k-NN Hyperparameter Tuning")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "knn_hyperparameter_tuning.png"), dpi=300)
    plt.close()

    # Retrain on full training data (train + val)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    best_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    t0 = time.time()
    best_knn.fit(X_trainval, y_trainval)
    train_time_final = time.time() - t0

    t1 = time.time()
    y_test_pred = best_knn.predict(X_test)
    test_pred_time = time.time() - t1

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    return {
        "model": "k-NN",
        "best_k": best_k,
        "val_acc": max(val_accs),
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_time": train_time_final,
        "test_pred_time": test_pred_time,
        "y_test_pred": y_test_pred,
        "best_model": best_knn
    }


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def run_mlp_experiment(X_train, y_train, X_val, y_val, X_test, y_test, fig_dir):
    batch_size = 64
    model = MLP().to(device)

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)),
                            batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    early_stopper = EarlyStopping(patience=5, min_delta=1e-4)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    t0 = time.time()

    # Training loop
    for epoch in range(20):
        model.train()
        running_loss, running_correct, running_samples = 0.0, 0, 0

        # Iterate over batches
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y_batch).sum().item()
            running_samples += x_batch.size(0)

        # Evaluate on validation set
        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

        # Early stopping check
        if early_stopper.step(val_loss, model):
            print("Early stopping triggered.")
            break

    train_time = time.time() - t0

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    t1 = time.time()
    test_loss, test_acc, y_test_pred, y_test_true = evaluate(model, test_loader, criterion)
    test_pred_time = time.time() - t1
    test_f1 = f1_score(y_test_true, y_test_pred, average="macro")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mlp_loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLP Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mlp_accuracy_curve.png"), dpi=300)
    plt.close()

    return {
        "model": "MLP",
        "best_k": "",
        "val_acc": max(val_accs),
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_time": train_time,
        "test_pred_time": test_pred_time,
        "y_test_pred": y_test_pred,
        "best_model": model
    }


def save_confusion_matrices(y_test, y_test_pred_knn, y_test_pred_mlp, best_k, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_knn = confusion_matrix(y_test, y_test_pred_knn)
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=range(10))
    disp_knn.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"k-NN Confusion Matrix (k={best_k})")

    cm_mlp = confusion_matrix(y_test, y_test_pred_mlp)
    disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=range(10))
    disp_mlp.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("MLP Confusion Matrix")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"), dpi=300)
    plt.close()


def save_mlp_misclassifications(X_test, y_test, y_test_pred_mlp, fig_dir):
    mis_idx_mlp = np.where(y_test_pred_mlp != y_test)[0]
    num_show = min(16, len(mis_idx_mlp))

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.ravel()
    for i in range(16):
        ax = axes[i]
        if i < num_show:
            idx = mis_idx_mlp[i]
            ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"T:{y_test[idx]} P:{y_test_pred_mlp[idx]}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample MLP Misclassifications")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mlp_misclassifications.png"), dpi=300)
    plt.close()


def main():
    # Load and preprocess data
    data = load_and_prepare_data()

    # Generate visualizations
    plot_dataset_visuals(data["X_train_raw"], data["y_train_raw"], data["X_test_raw"], data["y_test_raw"],
                         out_dir=fig_dir)
    plot_tsne(data["X_train"], data["y_train"], out_dir=fig_dir)

    # Run experiments
    knn_res = run_knn_experiment(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"], fig_dir
    )
    mlp_res = run_mlp_experiment(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"], fig_dir
    )

    # Save evaluation figures
    save_confusion_matrices(data["y_test"], knn_res["y_test_pred"], mlp_res["y_test_pred"], knn_res["best_k"], fig_dir)
    save_mlp_misclassifications(data["X_test"], data["y_test"], mlp_res["y_test_pred"], fig_dir)

    # Save results to CSV
    results = pd.DataFrame([
        {
            "Model": "k-NN",
            "Best k": knn_res["best_k"],
            "Validation Accuracy": knn_res["val_acc"],
            "Test Accuracy": knn_res["test_acc"],
            "Test F1": knn_res["test_f1"],
            "Train Time (s)": knn_res["train_time"],
            "Test Prediction Time (s)": knn_res["test_pred_time"],
        },
        {
            "Model": "MLP",
            "Best k": "",
            "Validation Accuracy": mlp_res["val_acc"],
            "Test Accuracy": mlp_res["test_acc"],
            "Test F1": mlp_res["test_f1"],
            "Train Time (s)": mlp_res["train_time"],
            "Test Prediction Time (s)": mlp_res["test_pred_time"],
        }
    ])

    results.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)


if __name__ == "__main__":
    main()
