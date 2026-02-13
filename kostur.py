import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

with open("data.pickle", "rb") as f:
    data_dict = pickle.load(f)

spam_email_contents = data_dict["spam"]
ham_email_contents = data_dict["ham"]

spam_y = np.full(len(spam_email_contents), 1)
ham_y = np.full(len(ham_email_contents), 0)
y = np.concatenate([spam_y, ham_y])

X_train, X_test, y_train, y_test = train_test_split(spam_email_contents + ham_email_contents, y, test_size=0.3, random_state=69, stratify=y) 

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(model.predict(["Click this link for free Nigerian prince money: I need your credit card info!"]))





