"""
CLASSE PER VERIFICARE SE IL MODELLO FA DELLE PREVISIONI CORRETTE.
"""
import pytest
from Sentiment_Analysis.src.load_model import LoadModel
from Sentiment_Analysis.src.predict_sentiment import SentimentTester
from Sentiment_Analysis.src.evaluation import EvaluateModel
from datasets import load_dataset
import pandas as pd

@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    loader = LoadModel()
    tokenizer, model = loader.load_model()
    return tokenizer, model

@pytest.fixture(scope="module")
def df_external_dataset():
    dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "italian")
    return dataset["test"]

def test_model_can_predict_external_dataset(loaded_model_and_tokenizer, df_external_dataset):
    tokenizer, model = loaded_model_and_tokenizer
    tester = SentimentTester(model, tokenizer, df_external_dataset)

    df_results = tester.get_results_dataframe()

    # Verifico che tutte le righe siano state predette
    assert not df_results["Predetto"].isnull().any(), "Ci sono predizioni nulle"

def test_evaluation_metrics_external_dataset(loaded_model_and_tokenizer, df_external_dataset):
    tokenizer, model = loaded_model_and_tokenizer
    tester = SentimentTester(model, tokenizer, df_external_dataset)

    df_results = tester.get_results_dataframe()
    y_true = df_results["Reale"]
    y_pred = df_results["Predetto"]

    accuracy = (y_true == y_pred).mean()
    assert accuracy > 0.8, f"Accuracy troppo bassa sul dataset esterno: {accuracy:.2f}"
