import json
import logging
import pickle
import re
import string
from xgboost import XGBClassifier

import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data_cleaning import data_cleaning

file = "D:\VA\ChatbotCorpus.json"

logger = logging.getLogger()  # Initialzing Logger

logger = logging.getLogger("Intent_Classifier")

logger.info("Downloading the required nltk Packages")

nltk.download("punkt")
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("english")
nltk.download("wordnet")
wn = nltk.WordNetLemmatizer()


def query_preprocessing(query):
    query = re.sub("s+", " ", query)  # Cleaning the whitespaces in the text
    query = re.sub("[^-9A-Za-z ]", "", query)  # Removing the whitespace
    query = "".join(
        [i.lower() for i in query if i not in string.punctuation]
    )  # Removing Panctuations
    query = nltk.tokenize.word_tokenize(query)  # Tokenize the data
    query = [i for i in query if i not in stopwords]  # Removing Stop Words
    query = [wn.lemmatize(word) for word in query]  # Lemmatize the text
    query = " ".join(query)  # Converting list to string
    return query


def modelling(X_train_idf, Y_train):

    best_lr = LogisticRegression(n_jobs=1, C=1e5, max_iter=100000)

    best_rf = RandomForestClassifier(
        bootstrap=True,
        max_depth=70,
        max_features="auto",
        min_samples_leaf=4,
        min_samples_split=10,
        n_estimators=400,
    )

    best_xgb = XGBClassifier(
        colsample_bytree=0.3,
        gamma=0.0,
        learning_rate=0.3,
        max_depth=15,
        min_child_weight=3,
        n_estimators=2000,
    )

    eclf1 = VotingClassifier(
        estimators=[
            ("Logistic Regression", best_lr),
            ("Random Forest", best_rf),
            ("Xgboost", best_xgb),
        ],
        voting="soft",
    )

    eclf1 = eclf1.fit(X_train_idf, Y_train)

    with open("ensemble.pkl", "wb") as f:
        pickle.dump(eclf1, f)

    return eclf1


def TF_IDF():
    with open(file) as train_file:
        dict_train = json.load(train_file)

    # converting json dataset from dictionary to dataframe
    train = pd.DataFrame.from_dict(dict_train["sentences"])

    train = pd.DataFrame.from_dict(dict_train["sentences"])
    df = data_cleaning(train)
    X_train = df[df["training"] == True]["text"]
    Y_train = df[df["training"] == True]["intent"]
    X_test = df[df["training"] == False]["text"]
    Y_test = df[df["training"] == False]["intent"]
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
    return tfidfconverter.fit(X_train)


def predict(query, model, tfidfconverter):
    query = query_preprocessing(query)
    ectorizerized_entity = tfidfconverter.transform([query]).toarray()

    entity = model.predict(ectorizerized_entity)
    if entity[0] == 0:
        result = "DepartureTime"
    else:
        result = "FindConnection"

    return result
