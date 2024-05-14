from data_processing import *
from prolog import *
from utils import *
import os
from semantic_web import *

def main():
    # 1. CARICAMENTO DATASET CSV
    [local_df, file_path_dataset] = load_dataset()

    # 2. DATASET CLEANING
    #local_df = preprocess_data(local_df, "data/new_dataset.csv")  

    # 3.CREAZIONE KNOWLEDGE BASE
    #create_knowledge_base(local_df)

    # 4. INFERENTIAL REASONING
    file_path_new_dataset = inference_data(file_path_dataset)

    # 5. WEB SEMANTICO    
    #add_height_from_semantic_web(file_path_new_dataset, file_path_new_dataset)



    # Fase 4: Addestramento del modello supervisionato
    #train_supervised_model()

    # Fase 5: Addestramento del modello non supervisionato
    #train_unsupervised_model()

    # Fase 6: Confronto dei modelli
    #compare_models()



if __name__ == "__main__":
    main()
