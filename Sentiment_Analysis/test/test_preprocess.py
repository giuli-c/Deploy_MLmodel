from Sentiment_Analysis.src.preprocess_data import PreprocessData

def test_preprocess():
    text = "Questo è un TEST! http://link.com"
    cleaned = PreprocessData.preprocess(text)
    assert "http" not in cleaned, "Il link non è stato rimosso"
    assert "!" not in cleaned, "La punteggiatura non è stata rimossa"