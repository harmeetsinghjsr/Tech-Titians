import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your data
# data = np.genfromtxt('emails.csv', delimiter=',', dtype=None)

# For the purpose of this example, let's create some dummy data
data = np.array([
    ("buy cheap viagra now", 1),
    ("low price for cialis", 1),
    ("limited time offer", 1),
    ("hello, how are you?", 0),
    ("meeting at 10am tomorrow", 0),
    ("can you help with the report?", 0),
])
emails = data[:, 0]
labels = data[:, 1].astype(int)

# Split the data into training and testing sets
emails_train, emails_test, labels_train, labels_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Convert the emails to a matrix of token counts
vectorizer = CountVectorizer()
emails_train_transformed = vectorizer.fit_transform(emails_train)
emails_test_transformed = vectorizer.transform(emails_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(emails_train_transformed, labels_train)

# Predict the labels for the test data
labels_pred = clf.predict(emails_test_transformed)

# Print the accuracy
print("Accuracy:", accuracy_score(labels_test, labels_pred))
# Load your data from a CSV file
data = pd.read_csv('spam.csv')

# Assuming the CSV file has two columns: 'email' and 'label'
emails = data['email']
labels = data['label']

# Split the data into training and testing sets
emails_train, emails_test, labels_train, labels_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Convert the emails to a matrix of token counts
vectorizer = CountVectorizer()
emails_train_transformed = vectorizer.fit_transform(emails_train)
emails_test_transformed = vectorizer.transform(emails_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(emails_train_transformed, labels_train)

# Predict the labels for the test data
labels_pred = clf.predict(emails_test_transformed)

# Print the accuracy
print("Accuracy:", accuracy_score(labels_test, labels_pred))