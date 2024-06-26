import csv
from pyswip import Prolog
from utils import addNewAttributes

# Funzione per scrivere fatti sul file prolog
def write_fact_to_file(fact, file_path_kb):
    # Verifica se il fatto è già presente
    with open(file_path_kb, 'r', encoding='utf-8') as file:
        existing_content = file.read()

    if fact not in existing_content:
        # Riapri il file in modalità append e scrivi il fatto
        with open(file_path_kb, 'a', encoding='utf-8') as file:
            file.write(f"{fact}.\n")

# Funzione per scrivere informazioni del giocatore nel file Prolog
def write_player_info(data_set,file_path_kb):

    with open(file_path_kb, "w", encoding="utf-8") as file:  # Overwrite the file (empty it)
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
            player_name = player_name.replace("'", "")
            squad = squad.replace("'","")
            prolog_fact = f"player_stats({player_id}, '{player_name}', '{nation}', '{position}', '{squad}',{min} ,{to_suc}, {rec}, {rec_prog}, {pas_tot_cmp}, {pas_ass}, {pass_cmp}, {pasProg}, {pasLonCmp})"            
            write_fact_to_file(prolog_fact, file_path_kb)

def write_rules(file_path_kb):  
    rules = """
        (strong_dribbler(Player) :- player_stats(_, Player, _, _, _, Min, ToSuc, _, _, _, _, _, _, _), ToSuc > 1.10, Min > 1000).
        (strong_playmaker(Player) :- player_stats(_, Player, _, _, _, Min, _, Rec, RecProg, PasTotCmp, PasAss, PasCmp, PasProg, PasLonCmp),  Min>1000, Rec>34, RecProg>3, PasTotCmp>33, PasAss>0.85, PasCmp>33, PasProg>3, PasLonCmp>3).
    """

    with open(file_path_kb, 'a') as file:
        file.write(rules)
    print("KB costruita")


def create_knowledge_base(local_df, file_path_kb):
    write_player_info(local_df,file_path_kb)
    write_rules(file_path_kb)


# funzione per effettuare ragionamento logico usando fatti e regole della KB
def inference_data(file_path_kb, file_path_dataset):
    list_playmakers = []
    list_dribblers = []

    # Crea un'istanza Prolog
    prolog = Prolog()
    prolog.consult(file_path_kb)

    # Query per i playmaker
    results = prolog.query("strong_playmaker(Player)")

    print("Forti playmakers:\n")
    for result in results:
        list_playmakers.append(result['Player'])
        print(result["Player"])

    # Query per i dribblatori
    results = prolog.query("strong_dribbler(Player)")

    print("\nForti dribblatori:\n")
    for result in results:
        list_dribblers.append(result['Player'])
        print(result["Player"])

    file_path_new_dataset = file_path_dataset.replace("dataset.csv","new_dataset.csv")
    #addNewAttributes(file_path_dataset, file_path_new_dataset, list_dribblers, list_playmakers)

