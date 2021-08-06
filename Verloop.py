import json
import logging
import pickle
import platform
import warnings
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

from data_cleaning import data_cleaning
from model_training import modelling
from model_training import predict

warnings.filterwarnings("ignore")


logger = logging.getLogger()  # Initialzing Logger

logger = logging.getLogger("Intent_Classifier")


file = "C:\Verloop_Assignment"  # Path to the File
try:
    if platform.system() == "Windows":
        RUN_IN_LOCAL = True
        # read data from local
        logger.info("Reading data from local ...")
        with open(file) as train_file:
            dict_train = json.load(train_file)

        # converting json dataset from dictionary to dataframe
        train = pd.DataFrame.from_dict(dict_train["sentences"])

    logger.info("data read successfully ...")


    df = data_cleaning(train)  # Calling the Data Cleaning Function

    logger.info("Spliting the Data into test and train ...")

    X_train = df[df["training"] == True]["text"]
    Y_train = df[df["training"] == True]["intent"]
    X_test = df[df["training"] == False]["text"]
    Y_test = df[df["training"] == False]["intent"]


    le = preprocessing.LabelEncoder()  # Initializing Label Encoder


    le.fit(Y_train)  # Fitting the Vectorizer with Train set
    Y_train = le.transform(Y_train)
    Y_test = le.transform(Y_test)

    logger.info("Creating TFIDF Vectorizer ...")

    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
    X_train_idf = tfidfconverter.fit_transform(X_trasin).toarray()
    X_test_idf = tfidfconverter.transform(X_test).toarray()

    feature_path = "Vectorizer.pkl"
    with open(feature_path, "wb") as fw:
        pickle.dump(tfidfconverter, fw)

    logger.info("Training Model ...")

    model = modelling(X_train_idf, Y_train)

    query = input("Enter Input Query:")

    intent = predict(query, model, tfidfconverter)

    print("The Intent of the Statement is {}".format(intent))
except Exception as e:
    print(e)
