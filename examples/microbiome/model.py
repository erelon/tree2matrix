import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=0.25)  # Dropout with probability 0.25
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(2, 3), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=0.25)  # Dropout with probability 0.25
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 61, 128)
        self.fc2 = nn.Linear(128, 2)  # Assuming binary classification
        self.dropout3 = nn.Dropout(p=0.5)  # Dropout with probability 0.5 for fully connected layer

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(-1, 64 * 61)  # Flattening before FC layers
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# Define train, evaluate, and test functions
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(data_loader.dataset)
    auc = roc_auc_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, auc, f1


if __name__ == '__main__':
    # Load data
    with open("results.pkl", "rb") as f:
        matrices_result, new_order_names = pickle.load(f)
    tags = pd.read_csv("tag_for_learning.csv")

    tags = np.array([int(i) for i in tags["Tag"]]).astype(np.int64)  # binary
    matrices_result = matrices_result[:, np.newaxis, :, :]  # 130, 1, 8, 743

    # Divide input into train and test data
    train_indices, test_indices = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(matrices_result, tags))
    train_data, train_labels = matrices_result[train_indices], np.array(tags)[train_indices]
    test_data, test_labels = matrices_result[test_indices], np.array(tags)[test_indices]

    # Initialize k-fold cross-validation
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0

    best_model = None
    best_val_accuracy = 0.0

    # Perform k-fold cross-validation
    models = []
    for train_indices, val_indices in k_fold.split(train_data, train_labels):
        fold += 1
        print(f"Fold {fold}:")
        train_fold_data, train_fold_labels = train_data[train_indices], train_labels[train_indices]
        val_fold_data, val_fold_labels = train_data[val_indices], train_labels[val_indices]

        # Create dataloaders
        train_fold_dataset = data.TensorDataset(torch.tensor(train_fold_data), torch.tensor(train_fold_labels))
        val_fold_dataset = data.TensorDataset(torch.tensor(val_fold_data), torch.tensor(val_fold_labels))

        train_fold_loader = data.DataLoader(train_fold_dataset, batch_size=32, shuffle=True)
        val_fold_loader = data.DataLoader(val_fold_dataset, batch_size=32)

        # Initialize model, loss, and optimizer
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001,
                               weight_decay=0.001)  # Adding weight decay for regularization

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss = train(model, train_fold_loader, criterion, optimizer)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")

        # Evaluation on validation set
        val_accuracy, _, _ = evaluate(model, val_fold_loader)
        print(f"Validation Accuracy: {val_accuracy}")

        # Check if this model has the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_model = model
            best_val_accuracy = val_accuracy
        models.append(model)

    # Now use the best model for evaluation on the test set
    test_dataset = data.TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))
    test_loader = data.DataLoader(test_dataset, batch_size=32)
    test_accuracy, test_auc, test_f1 = evaluate(best_model, test_loader)
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test AUC: {test_auc}")
    print(f"Test F1 Score: {test_f1}")

    all_acc, all_auc, all_f1 = [], [], []
    for m in models:
        test_accuracy, test_auc, test_f1 = evaluate(m, test_loader)
        all_acc.append(test_accuracy)
        all_auc.append(test_auc)
        all_f1.append(test_f1)
    print(f"Average Test Accuracy: {np.mean(all_acc)}")
    print(f"Average Test AUC: {np.mean(all_auc)}")
    print(f"Average Test F1 Score: {np.mean(all_f1)}")
    print(f"STD Test Accuracy: {np.std(all_acc)}")
    print(f"STD Test AUC: {np.std(all_auc)}")
    print(f"STD Test F1 Score: {np.std(all_f1)}")

    import pickle

    with open("results_model.pkl", "wb") as f:
        pickle.dump([all_acc, all_auc, all_f1], f)
