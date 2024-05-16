from data_processing import *
from PlayerStatsProlog import *
from utils import *
from semanticWeb import *
from bayesian_network import *
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer


def main():
    file_path_dataset = getFilePathDataSet("dataset.csv")
    print(file_path_dataset)
    file_path_kb = getFilePathKB()
    print(file_path_kb)

    # 1. CARICAMENTO DATASET CSV
    local_df = loadDataset("dataset.csv")
    # 2. DATASET CLEANING
    #local_df = preprocess_data(local_df, "data/new_dataset.csv")  

    # 3.CREAZIONE KNOWLEDGE BASE
    create_knowledge_base(local_df, file_path_kb)

    # 4. INFERENTIAL REASONING (inferenza di nuovi dati per aggiungere nuove feature al dataset)
    inference_data(file_path_kb, file_path_dataset)
    file_path_new_dataset = getFilePathDataSet("new_dataset.csv")

    # 5. WEB SEMANTICO    
    #add_height_from_semantic_web(file_path_new_dataset, file_path_new_dataset)
    
    # 6. BAYESIAN NETWORK
    # carico dataset aggiornato
    newDataset = loadDataset("new_dataset.csv")
    # rimuovo righe con valori nulli (perchè mi dava errore per righe con valori nulli)
    newDataset = newDataset.dropna()

    # Calcola il numero di righe da eliminare
    rows_to_drop = int(len(newDataset) * 0.70)

    # Seleziona casualmente il n% delle righe da eliminare
    rows_to_keep = newDataset.sample(n=len(newDataset) - rows_to_drop, random_state=42)

    # Crea un nuovo dataset con le sole righe selezionate
    newDataset = newDataset.loc[rows_to_keep.index]
    

    selected_columns = [
        'Pos', 'TouDefPen', 'TouDef3rd', 'TouMid3rd', 'TouAtt3rd', 
        'Tkl', 'PasProg', 'PPA', 'ScaDrib', 'Recov', 'height', 
        'dribbler', 'playmaker'
    ]

    newDataset = newDataset[selected_columns]
    #discretizzo il dataset
    # Explicitly set subsample to None to silence the warning
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform', subsample=None)

    # Select columns of float64 and int64 types
    continuos_columns = newDataset.select_dtypes(include=['float64', 'int64']).columns

    # Apply the discretizer to the selected columns
    newDataset[continuos_columns] = discretizer.fit_transform(newDataset[continuos_columns])

    print(newDataset)

    #Creo o leggo la rete bayesiana a seconda delle necessità
    #bayesianNetwork = create_BN(newDataset)
    bayesianNetwork = loadBayesianNetwork()

    #GENERAZIONE DI UN ESEMPIO RANDOMICO e PREDIZIONE DELLA SUA CLASSE
    esempioRandom = generateRandomExample(bayesianNetwork)
    print("ESEMPIO RANDOMICO GENERATO --->  ",esempioRandom)
    print("PREDIZIONE DEL SAMPLE RANDOM")
    predici(bayesianNetwork, esempioRandom.to_dict('records')[0], "Pos")
    
    # Fase 4: Addestramento del modello supervisionato
    #train_supervised_model()

    # Fase 5: Addestramento del modello non supervisionato
    #train_unsupervised_model()

    # Fase 6: Confronto dei modelli
    #compare_models()



if __name__ == "__main__":
    main()
