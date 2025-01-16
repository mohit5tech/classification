import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import classification_report
from datasets import load_dataset

class HuggingFaceDataset(Dataset):
    def __init__(self, dataset, label_map, transform=None):
        self.dataset = dataset
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = self.label_map[item['label']]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class MultiLabelClassifier:
    def __init__(self, dataset_name, img_size=(224, 224), batch_size=64):
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = None
        self.model = None

    def prepare_data(self):
        dataset = load_dataset(self.dataset_name)

        # Extract unique labels dynamically
        labels = sorted(set(dataset['train']['label']))
        label_map = {label: idx for idx, label in enumerate(labels)}
        
        train_dataset = dataset['train']
        val_dataset = dataset['validation'] if 'validation' in dataset else dataset['train'][:len(dataset['train'])//10]
        test_dataset = dataset['test'] if 'test' in dataset else dataset['train'][len(dataset['train'])//10:]

        self.classes = labels
        return train_dataset, val_dataset, test_dataset, label_map

    def create_dataloaders(self, train_dataset, val_dataset, test_dataset, label_map):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_loader = DataLoader(HuggingFaceDataset(train_dataset, label_map, transform), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(HuggingFaceDataset(val_dataset, label_map, transform), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(HuggingFaceDataset(test_dataset, label_map, transform), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def build_model(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))

        # Wrap the model for multi-GPU training
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def train_model(self, train_loader, val_loader, epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct = 0.0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data)

            train_loss /= len(train_loader.dataset)
            train_acc = train_correct.double() / len(train_loader.dataset)

            self.model.eval()
            val_loss, val_correct = 0.0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data)

            val_loss /= len(val_loader.dataset)
            val_acc = val_correct.double() / len(val_loader.dataset)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def evaluate_model(self, test_loader):
        self.model.eval()
        test_loss, test_correct = 0.0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels.data)

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct.double() / len(test_loader.dataset)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        return test_loss, test_acc

# Example usage
if __name__ == "__main__":
    print(f"Available GPUs: {torch.cuda.device_count()}")

    dataset_name = "danjacobellis/chexpert"  # Replace with Hugging Face dataset name

    classifier = MultiLabelClassifier(dataset_name)
    train_ds, val_ds, test_ds, label_map = classifier.prepare_data()
    train_loader, val_loader, test_loader = classifier.create_dataloaders(train_ds, val_ds, test_ds, label_map)
    classifier.build_model()
    classifier.train_model(train_loader, val_loader, epochs=10)
    classifier.evaluate_model(test_loader)
