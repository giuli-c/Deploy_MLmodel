import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from load_config import LoadConfig

def test_load_config():
    config = LoadConfig("Sentiment_Analysis/config.yaml").load_config()
    # Controllo che il file di configurazione contenga la chiave 'model'
    assert "model" in config, "Modello mancante nel file id configurazione!"
    # Controllo che il file di configurazione contenga la chiave 'training'
    assert "training" in config, "Parametri di training mancanti nel file id configurazione!"
    # Controllo che il file di configurazione contenga la chiave 'dataset'
    assert "dataset" in config, "Dataset mancante nel file id configurazione!"

def test_load_config_None():
    config = LoadConfig().load_config()
    assert config is not None, "Il file config.yaml non è stato caricato."
