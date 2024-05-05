import pandas as pd
from data_processing import *
from PlayersStatsProlog import *
from pyswip import Prolog
from utils import *
from utils import averageDribblingSuccess

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
    local_df = preprocess_data(local_df, "data/new_dataset.csv")  
    avgDribblingSuccess = averageDribblingSuccess(local_df)
    print(avgDribblingSuccess)
    selected_columns = ["Rk", "Player", "Nation", "Pos", "Squad", "ToSuc", "Rec", "RecProg", "PasTotCmp", "PasAss"]

    #csv_to_prolog('data/new_dataset.csv','src/kb.pl', selected_columns)

    # Query the knowledge base
    prolog = Prolog()
    # Load the Prolog file containing the knowledge base
    prolog.consult("src/kb.pl")


    
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
