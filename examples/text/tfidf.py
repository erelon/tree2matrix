import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

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

    # Step 3: TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Step 4: Train a multi-output classifier (Random Forest in this example)
    classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    classifier.fit(X_train_tfidf, y_train)

    # Step 5: Predict on the test set
    y_pred = classifier.predict(X_test_tfidf)

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
