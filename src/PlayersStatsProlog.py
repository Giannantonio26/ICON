import csv

def csv_to_prolog(csv_file, output_file, selected_columns):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        with open(output_file, 'w', encoding='utf-8') as out:
            for row in reader:
                # Generate Prolog facts for each row with selected columns
                prolog_fact = f"player_stats({', '.join([f'{row[k]}' for k in selected_columns])}).\n"
                out.write(prolog_fact)
