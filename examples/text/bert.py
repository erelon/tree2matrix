import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

if __name__ == '__main__':
    with open("develop.txt", "r") as f:
        raw_train_data = f.readlines()
    with open("topics.txt", "r") as f:
        topics = f.readlines()

    topics = topics[::2]
    topics = [i.strip() for i in topics]
    train_text_data = raw_train_data[2::4]
    train_topics = [i.strip(">\n").split()[2:] for i in raw_train_data[::4]]

    # Sample data: articles and their corresponding labels
    articles = train_text_data  # List of article texts
    textual_labels = train_topics  # List of lists, where each inner list contains the binary labels for an article

    # Step 1: Convert textual labels to binary labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(textual_labels)
    label_classes = mlb.classes_

    # Step 2: Split data into training and testing sets (if not already done)
    X_train, X_test, y_train, y_test = train_test_split(articles, binary_labels, test_size=0.2, random_state=42)

    # Function to tokenize text and get BERT embeddings
    def get_bert_embeddings(texts):
        encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        # Use the output of the [CLS] token as the fixed-size representation of the text
        return outputs.last_hidden_state[:, 0, :]

    # Step 3: Get BERT embeddings for training and testing data
    batch_size = 8  # Adjust batch size as needed
    train_embeddings = []
    for i in range(0, len(X_train), batch_size):
        batch_texts = X_train[i:i+batch_size]
        batch_embeddings = get_bert_embeddings(batch_texts)
        train_embeddings.append(batch_embeddings)
    X_train_bert = torch.cat(train_embeddings, dim=0)

    test_embeddings = []
    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test[i:i+batch_size]
        batch_embeddings = get_bert_embeddings(batch_texts)
        test_embeddings.append(batch_embeddings)
    X_test_bert = torch.cat(test_embeddings, dim=0)

    # Step 4: Train a multi-output classifier (Random Forest in this example)
    classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    classifier.fit(X_train_bert.cpu().numpy(), y_train)  # Move data to CPU for sklearn

    # Step 5: Predict on the test set
    y_pred = classifier.predict(X_test_bert.cpu().numpy())  # Move data to CPU for sklearn

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Print label classes
    print("Label Classes:", label_classes)

    # For multi-label classification, you may want to look at individual label performance
    # Here's an example of printing classification report for each label
    for i in range(len(label_classes)):
        print(f"Classification report for label {label_classes[i]}:")
        print(classification_report(y_test[:, i], y_pred[:, i]))
