import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pyspark.sql.dataframe import DataFrame
from joblibspark import register_spark

register_spark()

def default_transform():
    # Normalize CIFAR-10 images
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

class DeepCNN:
    def __init__(self,
                 num_classes: int = 10,
                 lr: float = 1e-3,
                 batch_size: int = 128,
                 epochs: int = 10,
                 device: str = None):
        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        # Build model, loss, optimizer
        self.model = self._build_model(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _build_model(self, num_classes: int) -> nn.Module:
        # Simple 3-layer CNN
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _prepare_loader(self, df: DataFrame, train: bool = True) -> DataLoader:
        # Collect images and labels from the Spark DataFrame
        X = np.array(df.select("image").collect()).astype(np.uint8)
        y = np.array(df.select("label").collect()).astype(np.int64)
        # Reshape flat arrays to (N,3,32,32)
        X = X.reshape(-1, 3, 32, 32)
        # Convert to floats and normalize
        X = X.astype(np.float32) / 255.0
        # Build torch tensors and dataset
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=train,
                          num_workers=4)

    def train(self, df: DataFrame) -> nn.Module:
        self.model.train()
        loader = self._prepare_loader(df, train=True)
        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"Epoch {epoch}/{self.epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

        return self.model

    def evaluate(self, df: DataFrame) -> List:
        self.model.eval()
        loader = self._prepare_loader(df, train=False)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        return [y_pred, acc, prec, rec, cm]
