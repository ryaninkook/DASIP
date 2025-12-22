#!/usr/bin/env python3
"""
Train CNN classifier for robot state classification from images.

This script trains a lightweight CNN on robot images to predict the
robot's state. Features include:
- Multiple CNN architectures (lightweight, MobileNet)
- Data augmentation
- Noise robustness testing
- Per-environment training and evaluation
- Visualization of training curves

Usage:
    # Train on all environments
    python train_cnn.py --data_path labels.csv --data_dir ../data
    
    # Train on specific environments
    python train_cnn.py --data_path labels.csv --envs square can lift
    
    # With noise robustness testing
    python train_cnn.py --data_path labels.csv --test_noise
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


# =============================================================================
# Constants
# =============================================================================

# Noise levels for robustness testing
NOISE_LEVELS = {
    "Clean": 0.0,
    "Low Noise": 0.05,
    "Medium Noise": 0.15,
    "High Noise": 0.30,
}

# All environments
ALL_ENVS = ["square", "tool_hang", "lift", "can", "transport", "pusht"]

# Label names
LABEL_NAMES = {
    0: "initialize",
    1: "near",
    2: "attempt",
    3: "holding",
    4: "pushing",
    5: "goal",
}


# =============================================================================
# Dataset
# =============================================================================

class RobotImageDataset(Dataset):
    """Dataset for robot state classification from images."""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        image_size: tuple = (32, 32),
        augment: bool = False,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.augment = augment
        
        # Ensure valid labels
        if "interaction_category" in self.dataframe.columns:
            self.dataframe["interaction_category"] = (
                self.dataframe["interaction_category"].fillna(0).astype(int)
            )
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["image_path"]
        
        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None:
                image = np.zeros((*self.image_size, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)
                
                # Histogram equalization for contrast
                for c in range(3):
                    image[:, :, c] = cv2.equalizeHist(image[:, :, c])
        except Exception:
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = int(row["interaction_category"])
        
        return image, label


# =============================================================================
# Models
# =============================================================================

class LightweightCNN(nn.Module):
    """Lightweight CNN for fast inference."""
    
    def __init__(self, num_classes: int = 6, input_size: int = 32):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate flattened size
        self.flat_size = 128 * (input_size // 8) ** 2
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        self._init_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeeperCNN(nn.Module):
    """Deeper CNN with residual-like connections."""
    
    def __init__(self, num_classes: int = 6, input_size: int = 32):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Training Functions
# =============================================================================

def create_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 32,
):
    """Create training and test dataloaders with balanced sampling."""
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = RobotImageDataset(
        train_df, transform=train_transform, image_size=(image_size, image_size)
    )
    test_dataset = RobotImageDataset(
        test_df, transform=test_transform, image_size=(image_size, image_size)
    )
    
    # Weighted sampling for class imbalance
    class_counts = train_df["interaction_category"].value_counts().sort_index()
    weights = 1.0 / class_counts.values
    sample_weights = train_df["interaction_category"].map(
        lambda x: weights[x] if x < len(weights) else 1.0
    ).values
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        "loss": total_loss / len(loader),
        "accuracy": 100.0 * correct / total,
        "predictions": all_preds,
        "labels": all_labels,
    }


def evaluate_with_noise(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    noise_levels: dict = None,
):
    """Evaluate model with different noise levels."""
    noise_levels = noise_levels or NOISE_LEVELS
    results = {}
    
    model.eval()
    
    for noise_name, noise_std in noise_levels.items():
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                # Add Gaussian noise
                if noise_std > 0:
                    noise = torch.randn_like(images) * noise_std
                    images = images + noise
                    images = torch.clamp(images, 0, 1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        results[noise_name] = {
            "loss": total_loss / len(loader),
            "accuracy": 100.0 * correct / total,
            "predictions": all_preds,
            "labels": all_labels,
        }
        
        print(f"  {noise_name}: {results[noise_name]['accuracy']:.2f}%")
    
    return results


# =============================================================================
# Main Training
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    patience: int = 10,
    save_path: str = None,
):
    """Full training loop with early stopping."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    
    best_accuracy = 0
    patience_counter = 0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_results = evaluate(model, test_loader, criterion, device)
        test_loss = test_results["loss"]
        test_acc = test_results["accuracy"]
        
        # Log
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%"
        )
        
        # LR scheduling
        scheduler.step(test_acc)
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            
            if save_path:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "best_accuracy": best_accuracy,
    }


def plot_training_curves(
    history: dict,
    save_path: str,
    title: str = "Training Curves",
):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["train_losses"], label="Train")
    ax1.plot(history["test_losses"], label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    
    ax2.plot(history["train_accs"], label="Train")
    ax2.plot(history["test_accs"], label="Test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN classifier for robot state classification"
    )
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--data_dir", type=str, default="../data", help="Image directory")
    parser.add_argument("--output_dir", type=str, default="cnn_models", help="Output directory")
    parser.add_argument("--envs", type=str, nargs="+", default=None, help="Environments to train")
    parser.add_argument("--model", type=str, choices=["light", "deep"], default="light")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test_noise", action="store_true", help="Test noise robustness")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Filter to valid samples with images
    df = df.dropna(subset=["image_path", "interaction_category"])
    df = df[df["image_path"].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]
    print(f"Found {len(df)} valid samples with images")
    
    # Determine environments
    envs = args.envs or df["env_choice"].unique().tolist()
    envs = [e for e in envs if e in df["env_choice"].values]
    
    print(f"Training on environments: {envs}")
    
    # Results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = output_dir / f"results_{timestamp}.txt"
    
    for env in envs:
        print(f"\n{'='*60}")
        print(f"Training on: {env}")
        print("=" * 60)
        
        # Filter data
        env_df = df[df["env_choice"] == env]
        
        if len(env_df) < 50:
            print(f"Skipping {env}: only {len(env_df)} samples")
            continue
        
        # Split data
        train_df, test_df = train_test_split(
            env_df, test_size=0.2, random_state=42,
            stratify=env_df["interaction_category"]
        )
        
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            train_df, test_df,
            batch_size=args.batch_size,
        )
        
        # Create model
        num_classes = int(df["interaction_category"].max() + 1)
        if args.model == "light":
            model = LightweightCNN(num_classes=num_classes).to(device)
        else:
            model = DeeperCNN(num_classes=num_classes).to(device)
        
        # Train
        model_path = output_dir / f"{env}_cnn.pth"
        history = train_model(
            model, train_loader, test_loader, device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            save_path=str(model_path),
        )
        
        # Plot curves
        plot_training_curves(
            history,
            str(output_dir / "visualizations" / f"{env}_curves.png"),
            title=env,
        )
        
        # Noise testing
        if args.test_noise:
            print(f"\nNoise robustness testing for {env}:")
            # Load best model
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            noise_results = evaluate_with_noise(
                model, test_loader, device,
                nn.CrossEntropyLoss(),
            )
            
            all_results[env] = {
                "best_accuracy": history["best_accuracy"],
                "noise_results": noise_results,
            }
        else:
            all_results[env] = {"best_accuracy": history["best_accuracy"]}
        
        print(f"\nBest accuracy for {env}: {history['best_accuracy']:.2f}%")
    
    # Save summary
    with open(results_file, "w") as f:
        f.write(f"CNN Training Results - {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        
        for env, results in all_results.items():
            f.write(f"{env}: {results['best_accuracy']:.2f}%\n")
            
            if "noise_results" in results:
                for noise_name, noise_data in results["noise_results"].items():
                    f.write(f"  {noise_name}: {noise_data['accuracy']:.2f}%\n")
        
        f.write("\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {output_dir}/")


if __name__ == "__main__":
    main()

