import csv
from pyswip import Prolog
from utils import add_new_attributes
# Function to write fact to the Prolog file
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

def write_rules():  
    rules = """
        (strong_dribbler(Player) :- player_stats(_, Player, _, _, _, Min, ToSuc, _, _, _, _, _, _, _), ToSuc > 1.10, Min > 1000).
        (strong_playmaker(Player) :- player_stats(_, Player, _, _, _, Min, _, Rec, RecProg, PasTotCmp, PasAss, PasCmp, PasProg, PasLonCmp),  Min>1000, Rec>34, RecProg>3, PasTotCmp>33, PasAss>0.85, PasCmp>33, PasProg>3, PasLonCmp>3).
    """

    # Append the string to a Prolog file
    with open('your_prolog_file.pl', 'a') as file:
        file.write(rules)


def create_knowledge_base(local_df):
    write_player_info(local_df)
    write_rules()

def inference_data(file_path_dataset):
    list_playmakers = []
    list_dribblers = []

    # Create a Prolog instance
    prolog = Prolog()

    # Consult the Prolog file containing the player facts
    prolog.consult("src/kb.pl")

    # Query for strong playermer
    results = prolog.query("strong_playmaker(Player)")

    # Print the results
    print("Strong playmakers:\n")
    for result in results:
        list_playmakers.append(result['Player'])
        #print(result["Player"])

    # Query for strong dribblers
    results = prolog.query("strong_dribbler(Player)")

    # Print the results
    print("\n\nStrong dribblers:\n")
    for result in results:
        list_dribblers.append(result['Player'])
        #print(result["Player"])

    file_path_new_dataset = file_path_dataset.replace("dataset.csv","new_dataset.csv")
    add_new_attributes(file_path_dataset, file_path_new_dataset, list_dribblers, list_playmakers)

    return file_path_new_dataset

