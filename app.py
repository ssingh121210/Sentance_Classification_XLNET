import os
import pickle
import threading
import warnings

import joblib
import uvicorn
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


from model_training import TF_IDF, query_preprocessing

# Initialize a app

app = FastAPI()
counter = 0
lock = threading.Lock()

path = "D:/VA/model/"


# Model
with open(path + "ensemble.pkl", "rb") as f:
    entity_cls = pickle.load(f)


# ML Pred


@app.get("/predict_intent/{query}")
async def predict(query):
    """
    Prediction function to post the result of the
    model output

    Args:
        query ([str]): Input Sentance

    Returns:
        [json]: With the Request id and the Output of the model
    """
    global counter
    query = query_preprocessing(query)
    tf1_new = TF_IDF()
    vectorizerized_entity = tf1_new.transform([query]).toarray()
    prrediction = entity_cls.predict(vectorizerized_entity)
    if prrediction[0] == 0:
        result = "DepartureTime"
    else:
        result = "FindConnection"

    with lock:
        counter += 1

    return {"reques_id": counter, "intent": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
