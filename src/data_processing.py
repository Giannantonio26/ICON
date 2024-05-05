import pandas as pd

def preprocess_data(df, output_file):
    # Verifica la presenza di valori mancanti
    print(df.isnull().sum())
    # Elimina le righe con valori mancanti
    df = df.dropna()
    # Rimuovi eventuali duplicati
    df = df.drop_duplicates() 
    # Lista di attributi da rimuovere
    set_pieces = ['TI', 'CK', 'CkIn', 'CkOut', 'CkStr', 'PasCrs']
    miscellaneous = ['SCA', 'GCA', 'Recov', 'AerWon', 'AerLost', 'AerWon%']
    discipline = ['CrdY', 'CrdR', '2CrdY', 'Fls', 'Off']

    # Combinazione di tutti gli attributi da rimuovere
    attributes_to_remove = set_pieces + miscellaneous + discipline

    # Eliminazione delle colonne dal dataset
    df = df.drop(columns=attributes_to_remove)

    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file, index=False)



