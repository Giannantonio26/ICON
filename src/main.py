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
    [newDataset] = loadDataset("new_dataset.csv")
    bNetCreation(newDataset)


    # Fase 4: Addestramento del modello supervisionato
    #train_supervised_model()

    # Fase 5: Addestramento del modello non supervisionato
    #train_unsupervised_model()

    # Fase 6: Confronto dei modelli
    #compare_models()



if __name__ == "__main__":
    main()
