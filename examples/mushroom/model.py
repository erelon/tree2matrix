import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models


class MushroomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])


class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        # Load a pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)
        # Remove the last layer (fully connected layer)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        # Modify the first layer to accept grayscale images
        self.resnet_features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Define additional layers for classification
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)  # Output has 2 classes

    def forward(self, x):
        x = self.resnet_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Load data and labels
    tags = pd.read_csv('mushroom.csv')["class"]
    tags = np.array([i == "p" for i in tags]).astype(np.int64)  # binary

    with open("results.pkl", "rb") as f:
        matrices_result, new_order_names = pickle.load(f)

    matrices_result = matrices_result[:, np.newaxis, :, :]  # 61069, 1, 39, 119

    # Split data into training, validation, and test sets randomly
    train_data, test_val_data, train_labels, test_val_labels = train_test_split(matrices_result, tags, test_size=0.3,
                                                                                random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(test_val_data, test_val_labels, test_size=0.5,
                                                                    random_state=42)

    # Create datasets and dataloaders
    train_dataset = MushroomDataset(train_data, train_labels)
    val_dataset = MushroomDataset(val_data, val_labels)
    test_dataset = MushroomDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = ModifiedResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 4
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device).squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(val_loader):
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device).squeeze()).sum().item()
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Accuracy on validation set: {100 * correct / total:.2f}%")

    # Test loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader):
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device).squeeze()).sum().item()
    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
