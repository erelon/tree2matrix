import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier

if __name__ == '__main__':
    # Load data
    data = pd.read_csv("sub_pca_tax_7_log_rare_bact_5_without_viruses.csv")
    tags = pd.read_csv("tag_for_learning.csv")

    del data["Unnamed: 0"]
    tags = np.array([int(i) for i in tags["Tag"]]).astype(np.int64)  # binary

    # Divide input into train and test data
    train_data, test_data, train_labels, test_labels = train_test_split(data, tags, test_size=0.2,
                                                                        random_state=42)

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
        train_fold_data, train_fold_labels = train_data.iloc[train_indices], train_labels[train_indices]
        val_fold_data, val_fold_labels = train_data.iloc[val_indices], train_labels[val_indices]

        # Initialize TabNet model
        clf = TabNetClassifier()

        # Fit model
        clf.fit(
            X_train=train_fold_data.values,
            y_train=train_fold_labels,
            eval_set=[(val_fold_data.values, val_fold_labels)],
            patience=10,  # Early stopping rounds
            max_epochs=1000,  # Maximum number of epochs
            eval_metric=['accuracy'],  # Evaluation metric
            loss_fn=torch.nn.CrossEntropyLoss(),  # Loss function
        )

        # Print keys in clf.history
        print(clf.history.history["val_0_accuracy"])

        # Evaluation on validation set
        # Update the key for validation accuracy according to the printed keys
        val_accuracy = clf.history.history["val_0_accuracy"][-1]  # Replace 'valid_accuracy' with the correct key
        print(f"Validation Accuracy: {val_accuracy}")

        # Check if this model has the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_model = clf
            best_val_accuracy = val_accuracy
        models.append(clf)

    # Now use the best model for evaluation on the test set
    test_preds = best_model.predict(test_data.values)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test AUC: {test_auc}")
    print(f"Test F1 Score: {test_f1}")

    all_acc, all_auc, all_f1 = [], [], []
    for m in models:
        test_preds = m.predict(test_data.values)
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_auc = roc_auc_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds)

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

    with open("results_tabnet.pkl", "wb") as f:
        pickle.dump([all_acc, all_auc, all_f1], f)

