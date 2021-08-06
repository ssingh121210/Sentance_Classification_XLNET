import logging
import re
import string

import nltk

logger = logging.getLogger()  # Initialzing Logger

logger = logging.getLogger("Intent_Classifier")


nltk.download("punkt")
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("english")
nltk.download("wordnet")
wn = nltk.WordNetLemmatizer()


def data_cleaning(df):
    """Cleaning the Text in the Data frame by removing stop words,numbers,tokenizing
    ,removing panctuations.

    Args:
        df ([DataFrame]): DataFrame with text and intent

    Returns:
        [DataFrame]: Cleaned DataFrame
    """
    logger.info("Data Cleaning started ...")
    for x in range(len(df)):
        df["text"][x] = re.sub(
            "s+", " ", df["text"][x]
        )  # Cleaning the whitespaces in the text
        df["text"][x] = re.sub(
            "[^-9A-Za-z ]", "", df["text"][x]
        )  # Removing the whitespace
        df["text"][x] = "".join(
            [i.lower() for i in df["text"][x] if i not in string.punctuation]
        )  # Removing Panctuations
        df["text"][x] = nltk.tokenize.word_tokenize(df["text"][x])  # Tokenize the data
        df["text"][x] = [
            i for i in df["text"][x] if i not in stopwords
        ]  # Removing Stop Words
        df["text"][x] = [
            wn.lemmatize(word) for word in df["text"][x]
        ]  # Lemmatize the text
        df["text"][x] = " ".join(df["text"][x])  # Converting list to string
        logger.info("Data Cleaning Done ...")

    return df
