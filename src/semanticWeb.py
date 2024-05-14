from SPARQLWrapper import SPARQLWrapper, JSON
import csv

def retrieve_player_height(player_uri):
    height = 0
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    # Construct the SPARQL query
    query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>

        SELECT *
        WHERE {{
          <{player_uri}> a dbo:SoccerPlayer ;
                           dbo:height ?height .
        }}
    """

    sparql.setQuery(query)

    # Execute the query and process the results
    results = sparql.query().convert()
    if 'results' in results and 'bindings' in results['results']:
        for result in results['results']['bindings']:
            height = result['height']['value']
    else:
        print("No data found for the specified player.")

    return height

def add_height_from_semantic_web(input_file, output_file):
    # Open the input CSV file and read the data
    with open(input_file, mode='r', newline='', encoding='latin-1') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        fieldnames = reader.fieldnames  # Store the fieldnames before reading the data rows
        data = list(reader)
        for row in data:
            player_uri = "http://dbpedia.org/resource/"
            player_name = row['Player']
            player_name = player_name.replace(" ","_")
            player_uri += player_name
            height = retrieve_player_height(player_uri)
            print(height)

            if(height == 0):
                row['height'] = "null"
            else:
                row['height'] = height
        
    # Write the updated dataset with the new attribute to a new CSV file
    with open(output_file, mode='w', newline='', encoding='latin-1') as csv_file:
        print("FINALMENTE")
        fieldnames += ['height']  # Add the new attribute names to the fieldnames
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')

        # Write the header
        writer.writeheader()

        # Write each row with the new attribute
        for row in data:
            print(row['height'])
            writer.writerow(row)


    print("height added")
    