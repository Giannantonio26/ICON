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

    # Create a Prolog instance
    prolog = Prolog()

    # Define the Prolog facts
    prolog.assertz("player_stats(2689, 'Filip ?uri?i?', 'SRB', 'MFFW', 'Sampdoria', 1.09, 28.9, 3.2, 23.5, 1.84)")
    prolog.assertz("player_stats(1234, 'Another Player', 'Country', 'Position', 'Team', 0.6, 25.0, 2.5, 20.0, 1.5)")

    # Define the Prolog rule for strong dribblers
    prolog.assertz("(strong_dribbler(Player) :- player_stats(_, Player, _, _, _, ToSuc, _, _, _, _), ToSuc > 0.7293638392857144)")

    # Query for strong dribblers
    results = prolog.query("strong_dribbler(Player)")

    # Print the results
    print("Strong dribblers:")
    for result in results:
        print(result["Player"])




    # Print the results
    for result in results:
        print("Strong dribbler:", result["Player"])

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
