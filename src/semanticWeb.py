from SPARQLWrapper import SPARQLWrapper, JSON
import csv

def retrieve_player_height(player_uri):
    height = 0
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>

        SELECT *
        WHERE {{
          <{player_uri}> a dbo:SoccerPlayer ;
                           dbo:height ?height .
        }}
    """

    sparql.setQuery(query)

    results = sparql.query().convert()
    if 'results' in results and 'bindings' in results['results']:
        for result in results['results']['bindings']:
            height = result['height']['value']
    else:
        print("No data found for the specified player.")

    return height

def add_height_from_semantic_web(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='latin-1') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        fieldnames = reader.fieldnames
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
        
    with open(output_file, mode='w', newline='', encoding='latin-1') as csv_file:
        fieldnames += ['height']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')

        writer.writeheader()

        for row in data:
            print(row['height'])
            writer.writerow(row)


    print("altezza aggiunta")
    