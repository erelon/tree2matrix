import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, topics):
        self.data = data
        self.topics = topics

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'topic': self.topics[idx]}
        return sample


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_res = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # Removed padding
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.fc1 = nn.Linear(128 * 1 * 44, 64)  # Reduced fully connected layer size
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.bn3(x)

        residual = self.conv_res(residual)
        residual = self.pool(residual)
        x += residual
        x = F.relu(x)

        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = x.view(-1, 128 * 1 * 44)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))  # Sigmoid activation for multi-label classification
        return x


if __name__ == '__main__':

    with open("develop.txt", "r") as f:
        raw_train_data = f.readlines()
    train_topics = [i.strip(">\n").split()[2:] for i in raw_train_data[::4]]
    with open("topics.txt", "r") as f:
        topics = f.readlines()

    topics = topics[::2]
    topics = [i.strip() for i in topics]

    with open("results.pkl", "rb") as f:
        matrices_result, new_order_names = pickle.load(f)
    matrices_result = matrices_result[:, np.newaxis, :, :]


    def one_hot_encode(labels, num_classes):
        encoded_labels = np.zeros((len(labels), num_classes))
        for idx, label_list in enumerate(labels):
            for idl, label in enumerate(label_list):
                encoded_labels[idx, idl] = 1
        return encoded_labels


    train_topics = one_hot_encode(train_topics, num_classes=9)  # Assuming you have 9 classes

    matrices_result = torch.tensor(matrices_result).type(torch.float32).to("cuda:0")
    train_topics = torch.tensor(train_topics).type(torch.float32).to("cuda:0")
    # Split data into train, validation, and test sets
    data_train, data_valtest, topics_train, topics_valtest = train_test_split(matrices_result, train_topics,
                                                                              test_size=0.2,
                                                                              random_state=42)
    data_val, data_test, topics_val, topics_test = train_test_split(data_valtest, topics_valtest, test_size=0.5,
                                                                    random_state=42)

    # Create custom datasets
    train_dataset = CustomDataset(data_train, topics_train)
    val_dataset = CustomDataset(data_val, topics_val)
    test_dataset = CustomDataset(data_test, topics_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model

    # Initialize the model
    model = CNN(num_classes=9).to("cuda:0")

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0  # Initialize training loss
        for batch in train_loader:
            images, labels = batch['data'], batch['topic']
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # Accumulate training loss

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch['data'], batch['topic']
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.9).float()  # Thresholding the output probabilities to get binary predictions
                val_total += labels.size(0) * labels.size(1)  # Total number of labels
                val_correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch['data'], batch['topic']
            outputs = model(images)
            predicted = (outputs > 0.9).float()  # Thresholding the output probabilities to get binary predictions
            total += labels.size(0) * labels.size(1)  # Total number of labels
            correct += (predicted == labels).sum().item()

    print('Accuracy: {:.2f}%'.format(100 * correct / total))

    # Initialize empty lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate through the test set
    for batch in test_loader:
        images, labels = batch['data'], batch['topic']
        outputs = model(images)
        predicted = (outputs > 0.9).int()  # Thresholding the output probabilities to get binary predictions

        # Append true and predicted labels for each data point in the batch
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Compute multilabel confusion matrix
    conf_matrix = multilabel_confusion_matrix(true_labels, predicted_labels)

    # Print or visualize the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Initialize empty lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate through the test set
    for batch in test_loader:
        images, labels = batch['data'], batch['topic']
        outputs = model(images)
        predicted = (outputs > 0.9).int()  # Thresholding the output probabilities to get binary predictions

        # Append true and predicted labels for each data point in the batch
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    # For multi-label classification, you may want to look at individual label performance
    # Here's an example of printing classification report for each label
    for i in range(9):  # Assuming num_classes is defined
        print(f"Classification report for label {i}:")
        print(classification_report(true_labels[:, i], predicted_labels[:, i]))
