import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from load_model import LoadModel

def test_model_loading():
    loader = LoadModel()
    tokenizer, model = loader.load_model()
    assert tokenizer is not None and model is not None, "Modello e Tokenizer non sono stati caricati correttamente!"