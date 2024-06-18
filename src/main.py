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
    # rimuovo righe con valori nulli
    newDataset = newDataset.dropna()
    # Calcola il numero di righe da eliminare
    rows_to_drop = int(len(newDataset) * 0.90)
    # Seleziona casualmente le righe da mantenere
    rows_to_keep = newDataset.sample(n=len(newDataset) - rows_to_drop, random_state=40)
    # Crea un nuovo dataset con le sole righe selezionate
    newDataset = newDataset.loc[rows_to_keep.index]

    selected_columns = [
        'Pos', 'TouDefPen', 'TouDef3rd', 'TouMid3rd', 'TouAtt3rd', 
        'Tkl', 'PasProg', 'PPA', 'ScaDrib', 'Recov', 'height', 
        'dribbler', 'playmaker'
    ]

    newDataset = newDataset[selected_columns]
    #discretizzo il dataset
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform', subsample=None)
    # seleziona le colonne nel dataset che contengono valori di tipo float64 o int64 
    continuos_columns = newDataset.select_dtypes(include=['float64', 'int64']).columns
    # Applica il "discretizer" alle colonne selezionate
    newDataset[continuos_columns] = discretizer.fit_transform(newDataset[continuos_columns])
    print(newDataset)
    #Creazione o lettura della rete bayesiana in base alle necessità
    bayesianNetwork = create_BN(newDataset)
    #bayesianNetwork = loadBayesianNetwork()
    #GENERAZIONE DI UN ESEMPIO RANDOMICO e PREDIZIONE DEL RUOLO DI UN GIOCATORE
    esempioRandom = generateRandomExample(bayesianNetwork)
    print("ESEMPIO RANDOMICO GENERATO\n",esempioRandom)
    print("PREDIZIONE DEL SAMPLE RANDOM")
    predici(bayesianNetwork, esempioRandom.to_dict('records')[0], "Pos")
    
    # Fase 4: Addestramento del modello supervisionato
    #train_supervised_model()

    # Fase 5: Addestramento del modello non supervisionato
    #train_unsupervised_model()




if __name__ == "__main__":
    main()
