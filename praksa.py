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

print(len(spam_email_contents))
print(len(ham_email_contents))
