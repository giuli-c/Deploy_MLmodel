import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess_data import PreprocessData

def test_preprocess_replace_user_and_link():
    text = "@giulia questo è un link http://example.com"
    expected = "@user questo è un link http"
    cleaned = PreprocessData.preprocess(text)
    assert cleaned == expected, f"Risultato atteso: '{expected}', ottenuto: '{cleaned}'"

def test_preprocess_multiple_users_and_links():
    text = "Ciao @mario! Guarda http://site1.com e http://site2.it"
    expected = "Ciao @user! Guarda http http"
    cleaned = PreprocessData.preprocess(text)
    assert cleaned == expected, f"Risultato atteso: '{expected}', ottenuto: '{cleaned}'"


def test_preprocess_text_without_user_or_link():
    text = "Questo è un testo semplice senza utenti o link"
    expected = text  # Deve rimanere uguale
    cleaned = PreprocessData.preprocess(text)
    assert cleaned == expected, f"Il testo non doveva cambiare: '{cleaned}'"