from Sentiment_Analysis.src.load_config import LoadConfig

def test_load_config():
    config = LoadConfig().load_config()
    # Controllo che il file di configurazione contenga la chiave 'model'
    assert "model" in config, "Modello mancante nel file id configurazione!"
    # Controllo che il file di configurazione contenga la chiave 'training'
    assert "training" in config, "Parametri di training mancanti nel file id configurazione!"
    # Controllo che il file di configurazione contenga la chiave 'dataset'
    assert "dataset" in config, "Dataset mancante nel file id configurazione!"
