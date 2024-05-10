import pandas as pd
from data_processing import *
from PlayersStatsProlog import *
from pyswip import Prolog
from utils import *
import os
from semantic_web import *

def main():
    #carica il dataset CSV
    file_separator = os.path.sep
    current_directory = os.getcwd()
    if("PROGETTO_ICON"+file_separator+"ICON" in current_directory):
        file_path_dataset = "data"+file_separator+"dataset.csv"
    else:
        file_path_dataset = "ICON"+file_separator+"data"+file_separator+"dataset.csv"

    # Leggi il file CSV senza specificare l'encoding
    try:
        local_df = pd.read_csv(file_path_dataset,sep=";")
    except UnicodeDecodeError:
        local_df = pd.read_csv(file_path_dataset, encoding='latin-1',sep=";")
    if local_df is None:
        print("Errore nel caricamento del file CSV.")
    else:
        print("ok")
    # preprocess dei dati
    #local_df = preprocess_data(local_df, "data/new_dataset.csv")  

    #write_player_info(local_df)

    list_playmakers = []
    list_dribblers = []

    # Create a Prolog instance
    prolog = Prolog()

    # Consult the Prolog file containing the player facts
    prolog.consult("src/kb.pl")

    # Query for strong playermer
    results = prolog.query("strong_playmaker(Player)")

    # Print the results
    print("Strong playmakers:\n")
    for result in results:
        list_playmakers.append(result['Player'])
        #print(result["Player"])

    # Query for strong dribblers
    results = prolog.query("strong_dribbler(Player)")

    # Print the results
    print("\n\nStrong dribblers:\n")
    for result in results:
        list_dribblers.append(result['Player'])
        #print(result["Player"])

    # CREA ATTRIBUTO dribbler nel dataset csv (imposta a true/false usando  giocatori ritornati dalla query)
    # CREA ATTRIBUTO playmaker nel dataset csv (imposta a true/false usando  giocatori ritornati dalla query)
    file_path_new_dataset= file_path_dataset.replace("dataset.csv","new_dataset.csv")
    add_new_attributes(file_path_dataset, file_path_new_dataset, list_dribblers, list_playmakers)

    # Fase 2: #Aggiungi al dataset attributi prelevati dal web semantico
        
    add_height_from_semantic_web(file_path_new_dataset, file_path_new_dataset)

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
