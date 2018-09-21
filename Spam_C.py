from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_table('SMSSpamCollection.CSV', sep='\t', header=None, names=['label', 'sms_message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)
count_vector = CountVectorizer(stop_words='english')

#print(count_vector)
print(df.head())
Training_data = count_vector.fit_transform(X_train)
Test_data = count_vector.transform(X_test)
naive_bayes = MultinomialNB().fit(Training_data, y_train)
predictions = naive_bayes.predict(Test_data)
print(accuracy_score(y_test, predictions))
print(precision_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(f1_score(y_test, predictions))
