import pandas as pd

def preprocess_data(df, output_file):
    # Verifica la presenza di valori mancanti
    print(df.isnull().sum())
    # Elimina le righe con valori mancanti
    df = df.dropna()
    # Rimuovi eventuali duplicati
    df = df.drop_duplicates() 
    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file, index=False)

    return df

