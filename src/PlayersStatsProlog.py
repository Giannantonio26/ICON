import csv

# Function to convert CSV to Prolog facts
def csv_to_prolog(csv_file, output_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        with open(output_file, 'w', encoding='utf-8') as out:  # Specify UTF-8 encoding for writing
            for row in reader:
                # Generate Prolog facts for each row
                prolog_fact = f"player_stats({', '.join([f'{k}={v}' for k, v in row.items()])}).\n"
                out.write(prolog_fact)
