import pandas as pd
from data_processing import *
from PlayersStatsProlog import *
from pyswip import Prolog
from utils import *

def main():
    #carica il dataset CSV
    file_path = 'data/dataset.csv'
    # Leggi il file CSV senza specificare l'encoding
    try:
        local_df = pd.read_csv(file_path,sep=";")
    except UnicodeDecodeError:
        local_df = pd.read_csv(file_path, encoding='latin-1',sep=";")
    if local_df is None:
        print("Errore nel caricamento del file CSV.")
    else:
        print("ok")
    # preprocess dei dati
    #local_df = preprocess_data(local_df, "data/new_dataset.csv")  

    #write_player_info(local_df)

    # Create a Prolog instance
    prolog = Prolog()

    # Consult the Prolog file containing the player facts
    prolog.consult("src/kb.pl")
    # Query for strong dribblers
    results = prolog.query("strong_playmaker(Player)")

    # Print the results
    print("Strong playmakers:\n")
    for result in results:
        print(result["Player"])

    # Query for strong dribblers
    results = prolog.query("strong_dribbler(Player)")

    # Print the results
    print("\n\nStrong dribblers:\n")
    for result in results:
        print(result["Player"])

    # CREA ATTRIBUTO strong_dribbler nel dataset csv (imposta a true/false usando  giocatori ritornati dalla query)
    # CREA ATTRIBUTO strong_playmaker nel dataset csv (imposta a true/false usando  giocatori ritornati dalla query)


    # Fase 2: #Aggiungi al dataset attributi prelevati dal web semantico
    #run_semantic_integration()


    # Fase 3: Ragionamento relazionale
    #perform_relational_reasoning()

    # Fase 4: Addestramento del modello supervisionato
    #train_supervised_model()

    # Fase 5: Addestramento del modello non supervisionato
    #train_unsupervised_model()

    # Fase 6: Confronto dei modelli
    #compare_models()



if __name__ == "__main__":
    main()
