import csv


def write_fact_to_file(fact, file_path):
    # Verifica se il fatto è già presente
    with open(file_path, 'r', encoding='utf-8') as file:
        existing_content = file.read()

    if fact not in existing_content:
        # Riapri il file in modalità append e scrivi il fatto
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f"{fact}.\n")

# Function to write player information to the Prolog file
def write_player_info(data_set):
    file_path = "src/kb.pl"
    with open(file_path, "w", encoding="utf-8") as file:  # Overwrite the file (empty it)
        for index, row in data_set.iterrows():
            player_id = row['Rk']
            player_name = row['Player']
            nation = row['Nation']
            position = row['Pos']
            squad = row['Squad']
            min = row['Min']
            to_suc = row['ToSuc']
            rec = row['Rec']
            rec_prog = row['RecProg']
            pas_tot_cmp = row['PasTotCmp']
            pas_ass = row['PasAss']
            pass_cmp = row['PasCmp']
            pasProg = row['PasProg']
            pasLonCmp = row['PasLonCmp']

            # Remove any single quotes from player_name
            player_name = player_name.replace("'", "")
            squad = squad.replace("'","")
            # Construct the Prolog fact for the player
            prolog_fact = f"player_stats({player_id}, '{player_name}', '{nation}', '{position}', '{squad}',{min} ,{to_suc}, {rec}, {rec_prog}, {pas_tot_cmp}, {pas_ass}, {pass_cmp}, {pasProg}, {pasLonCmp})"
            
            # Write the Prolog fact to the file
            write_fact_to_file(prolog_fact, file_path)
