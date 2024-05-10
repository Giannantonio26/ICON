import csv

def add_new_attributes(input_file, output_file, list_dribblers, list_playmakers):
    # Open the input CSV file and read the data
    with open(input_file, mode='r', newline='', encoding='latin-1') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        fieldnames = reader.fieldnames  # Store the fieldnames before reading the data rows
        data = list(reader)
        for row in data:
            if row['Player'] in list_playmakers:
                playmaker_value = True
            else:
                playmaker_value = False
            row["playmaker"] = playmaker_value

            if row['Player'] in list_dribblers:
                dribbler_value = True
            else:
                dribbler_value = False
            row["dribbler"] = dribbler_value
           
    # Write the updated dataset with the new attribute to a new CSV file
    with open(output_file, mode='w', newline='', encoding='latin-1') as csv_file:
        fieldnames += ['dribbler', 'playmaker']  # Add the new attribute names to the fieldnames
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')

        # Write the header
        writer.writeheader()

        # Write each row with the new attribute
        for row in data:
            writer.writerow(row)


    print("New attribute added and dataset saved.")



def averageRec(df):
    rec = df['Rec'].mean()
    print(rec)
    return rec

def averageRec_prog(df):
    rec_prog = df['RecProg'].mean()
    print(rec_prog)
    return rec_prog

def averagePas_tot_cmp(df):
    pas_tot_cmp = df['PasTotCmp'].mean()
    print(pas_tot_cmp)

    return pas_tot_cmp

def averagePas_pas_ass(df):
    pas_ass = df['PasAss'].mean()
    print(pas_ass)

    return pas_ass

def averagePas_pass_cmp(df):
    pass_cmp = df['PasCmp'].mean()
    print(pass_cmp)

    return pass_cmp

def averagePas_pas_prog(df):
    pas_prog = df['PasProg'].mean()
    print(pas_prog)
    return pas_prog

def averagePas_pasTotCmp(df):
    pasTotCmp = df['PasTotCmp'].mean()
    print(pasTotCmp)
    return pasTotCmp

def averagePas_pasLonCmp(df):
    pasLonCmp = df['PasLonCmp'].mean()
    print(pasLonCmp)
    return pasLonCmp



