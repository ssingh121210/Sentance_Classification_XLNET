import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from model_training import query_preprocessing


path = "D:/VA/model/"
with open(path + "entity_vectorizer.pkl", "rb") as f:
    entity_vectorizer = pickle.load(f)
    tf1_new = TfidfVectorizer(
        max_features=2000, min_df=5, max_df=0.7, vocabulary=entity_vectorizer
    )

query = input("Query")

query = query_preprocessing(query)
vectorizerized_entity = tf1_new.fit_transform([query]).toarray()
print(vectorizerized_entity)
