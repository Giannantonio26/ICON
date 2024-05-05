import pandas as pd

def preprocess_data(df, output_file):
    # Verifica la presenza di valori mancanti
    print(df.isnull().sum())
    # Elimina le righe con valori mancanti
    df = df.dropna()
    # Rimuovi eventuali duplicati
    df = df.drop_duplicates() 
    # Lista di attributi da rimuovere
    desired_columns = [
    'Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'MP', 'Starts',
    'Min', '90s', 'Goals', 'Shots', 'SoT', 'G/Sh', 'G/SoT', 'ShoDist', 'ShoFK', 'ShoPK',
    'PasTotCmp', 'PasTotAtt', 'PasTotCmp%', 'PasShoCmp', 'PasShoAtt', 'PasShoCmp%',
    'PasMedCmp', 'PasMedAtt', 'PasMedCmp%', 'PasLonCmp', 'PasLonAtt', 'PasLonCmp%',
    'Assists', 'PasAss', 'Pas3rd', 'PPA', 'CrsPA', 'PasProg', 'PasAtt', 'TB', 'Sw',
    'PasCrs', 'CK', 'PasCmp', 'PasBlocks', 'SCA', 'ScaPassLive', 'ScaPassDead', 'ScaDrib',
    'ScaSh', 'ScaFld', 'ScaDef', 'GCA', 'GcaPassLive', 'GcaPassDead', 'GcaDrib', 'GcaSh',
    'Tkl', 'TklDef3rd', 'TklMid3rd', 'TklAtt3rd', 'TklDri', 'TklDri%', 'TklDriPast',
    'Blocks', 'BlkSh', 'BlkPass', 'Int', 'Tkl+Int', 'Err', 'Touches', 'TouDefPen',
    'TouDef3rd', 'TouMid3rd', 'TouAtt3rd', 'TouAttPen', 'TouLive', 'ToAtt', 'ToSuc',
    'ToTkl', 'Carries', 'CarTotDist', 'CarPrgDist', 'CarProg', 'Car3rd', 'CPA', 'Rec',
    'RecProg', 'CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld', 'Crs', 'Recov', 'AerWon', 'AerLost'
    ]

    df = df[desired_columns]

    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file, index=False)

    return df

