import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader


class MushroomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])


if __name__ == '__main__':
    # Load data and labels
    raw = pd.read_csv('mushroom.csv')
    tags = raw["class"]
    data = raw
    del data["class"]
    tags = np.array([i == "p" for i in tags]).astype(np.int64)  # binary
    data_encoded = pd.get_dummies(data)

    # Split data into training, validation, and test sets randomly
    # data_to_use, loset_data, labels_to_use, losed_labels = train_test_split(data_encoded, tags, test_size=0.98, random_state=42)
    data_to_use, _, labels_to_use, _ = data_encoded, None, tags, None

    train_data, test_val_data, train_labels, test_val_labels = train_test_split(data_to_use, labels_to_use,
                                                                                test_size=0.3,
                                                                                random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(test_val_data, test_val_labels, test_size=0.5,
                                                                    random_state=42)

    # Define the TabNetClassifier model
    model = TabNetClassifier()

    # Fit the GridSearchCV object
    model.fit(
        X_train=train_data.values.astype(float),
        y_train=train_labels,
        eval_set=[(val_data.values, val_labels)],
        patience=5,  # Early stopping rounds
        max_epochs=1000,  # Maximum number of epochs
        eval_metric=['accuracy'],  # Evaluation metric
        loss_fn=torch.nn.CrossEntropyLoss(),  # Loss function
    )

    # Evaluate the best model on the test set
    test_predictions = model.predict(test_data.values.astype(float))
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Accuracy on test set: {100 * test_accuracy:.2f}%")
