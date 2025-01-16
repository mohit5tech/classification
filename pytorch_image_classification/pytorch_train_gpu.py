import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights  # Import the appropriate weights enum
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        filepath = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['labels']
        image = Image.open(filepath).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a LongTensor
        return image, label

class ChestXRayClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=64):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = None
        self.model = None

    def prepare_data(self):
        filepaths, labels = [], []

        folds = os.listdir(self.data_dir)
        for fold in folds:
            foldpath = os.path.join(self.data_dir, fold)
            filelist = os.listdir(foldpath)
            for file in filelist:
                filepaths.append(os.path.join(foldpath, file))
                labels.append(fold)

        df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
        df['labels'] = pd.Categorical(df['labels'])
        df['labels'] = df['labels'].cat.codes

        train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
        valid_df, test_df = train_test_split(dummy_df, train_size=0.6, shuffle=True, random_state=123)

        self.classes = pd.Categorical(df['labels']).categories
        return train_df, valid_df, test_df

    def create_dataloaders(self, train_df, valid_df, test_df):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = ChestXRayDataset(train_df, transform=transform)
        valid_dataset = ChestXRayDataset(valid_df, transform=transform)
        test_dataset = ChestXRayDataset(test_df, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader

    def build_model(self):
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # Use the weights parameter
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))

        # Use DataParallel to wrap the model for multi-GPU training
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def train_model(self, train_loader, valid_loader, epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct = 0.0, 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data)

                if (i + 1) % 10 == 0:  # Print logs every 10 batches
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            train_loss /= len(train_loader.dataset)
            train_acc = train_correct.double() / len(train_loader.dataset)

            self.model.eval()
            valid_loss, valid_correct = 0.0, 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    valid_correct += torch.sum(preds == labels.data)

            valid_loss /= len(valid_loader.dataset)
            valid_acc = valid_correct.double() / len(valid_loader.dataset)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

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

    def generate_classification_report(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds, target_names=self.classes)
        print(report)

# Example usage
if __name__ == "__main__":
    print(f"Available GPUs: {torch.cuda.device_count()}")

    data_dir = 'datasets/chest_xray_pneumonia/chest_xray/train'

    classifier = ChestXRayClassifier(data_dir)
    train_df, valid_df, test_df = classifier.prepare_data()
    train_loader, valid_loader, test_loader = classifier.create_dataloaders(train_df, valid_df, test_df)
    classifier.build_model()
    classifier.train_model(train_loader, valid_loader, epochs=10)
    classifier.evaluate_model(test_loader)
    classifier.generate_classification_report(test_loader)
