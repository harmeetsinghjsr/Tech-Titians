import tkinter as tk
import chardet
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.sparse import hstack

# Function to process text data
def process_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word not in stop_words]
    return ' '.join(words)

# Load dataset
with open('CN127me/spam.csv', 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']

raw_data_set = pd.read_csv('C:/Users/hs978/CodeSpace/CN127me/spam.csv', encoding=encoding)
data_set = raw_data_set.where((pd.notnull(raw_data_set)),'')

# Tokenize and preprocess text
data_set['processed_text'] = data_set['v2'].apply(process_text)

# Label encoding
data_set.loc[data_set['v1'] == 'spam', 'v1'] = 0
data_set.loc[data_set['v1'] == 'ham', 'v1'] = 1

# Split dataset
X = data_set['v2']
Y = data_set['v1']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

count_vectorizer = CountVectorizer()
X_train_word_freq = count_vectorizer.fit_transform(X_train)
X_test_word_freq = count_vectorizer.transform(X_test)

# Additional features
spam_keywords = ['free', 'buy now', 'limited time offer', 'click', 'warning', 'alert', 'suspicious']
def presence_of_keywords(text):
    return [1 if keyword in text else 0 for keyword in spam_keywords]

additional_features_train = X_train.apply(presence_of_keywords).apply(pd.Series)
additional_features_test = X_test.apply(presence_of_keywords).apply(pd.Series)

def email_length(text):
    return len(text)

X_train_email_length = X_train.apply(email_length)
X_test_email_length = X_test.apply(email_length)

# Combine features
X_train_combined = hstack([X_train_features, X_train_word_freq, additional_features_train, X_train_email_length.values.reshape(-1, 1)])
X_test_combined = hstack([X_test_features, X_test_word_freq, additional_features_test, X_test_email_length.values.reshape(-1, 1)])

# Model training
model = LogisticRegression()
model.fit(X_train_combined, Y_train)

svm_model = SVC()
svm_model.fit(X_train_combined, Y_train)

# GUI setup
def predict_and_display():
    new_email = text_input.get("1.0", tk.END).strip()
    processed_email = process_text(new_email)
    input_data_features = feature_extraction.transform([new_email])
    
    prediction_lr = model.predict(input_data_features)[0]
    prediction_svm = svm_model.predict(input_data_features)[0]

    if prediction_lr == 1:
        result_label_lr.config(text="Ham mail (LR)")
    else:
        result_label_lr.config(text="Spam mail (LR)")

    if prediction_svm == 1:
        result_label_svm.config(text="Ham mail (SVM)")
    else:
        result_label_svm.config(text="Spam mail (SVM)")

window = tk.Tk()
window.title("Spam Email Detector")

# Input area
tk.Label(window, text="Enter Email Text:").grid(row=0, column=0)
text_input = tk.Text(window)
text_input.grid(row=0, column=1)

# Button
tk.Button(window, text="Predict", command=predict_and_display).grid(row=1, column=1)

# Result labels
result_label_lr = tk.Label(window, text="")
result_label_lr.grid(row=2, column=0, columnspan=2)
result_label_svm = tk.Label(window, text="")
result_label_svm.grid(row=3, column=0, columnspan=2)

window.mainloop()
