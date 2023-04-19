from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import networkx as nx
import json

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/shortest_path', methods=['POST'])
def results():
    # Load data
    routes = pd.read_csv('Data/routes.csv', sep=',', header=None)
    routes = routes.drop([0, 1, 3, 5, 6, 7, 8], axis=1)
    routes.rename(columns={2: 'origin', 4: 'destination'}, inplace=True)
    isolated = ['BMY', 'GEA', 'ILP', 'KNQ', 'KOC', 'LIF', 'MEE', 'TGJ', 'TOU', 'UVE', 'ERS', 'MPA', 'NDU', 'OND',
                'BFI', 'CLM', 'ESD', 'FRD', 'AKB', 'DUT', 'IKO', 'KQA', 'SPB', 'SSB', 'CKX', 'TKJ', 'BLD', 'GCW']
    airports_df = pd.read_csv('Data/airports.dat', header=None,
                              names=['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude',
                                     'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'],
                              index_col=0)

    # Filter out isolated airports
    routes = routes[np.logical_not(routes.origin.isin(isolated))]
    routes = routes[np.logical_not(routes.destination.isin(isolated))]

    # Create graph
    llista1 = np.unique(routes.origin)
    llista2 = np.unique(routes.destination)
    airports = np.unique(np.concatenate([llista1, llista2], axis=0))
    routes_with_weight = routes.groupby(['origin', 'destination']).size()
    links = routes_with_weight.index.unique()
    weights = routes_with_weight.tolist()
    input_tuple = [link + (weight,) for link, weight in zip(links, weights)]
    g_und = nx.Graph()
    g_und.name = 'Undirected Graph'
    g_und.add_nodes_from(airports, bipartite=1)
    g_und.add_weighted_edges_from(input_tuple)

    # Find shortest path and save coordinates to JSON file
    source = request.form['source']
    destination = request.form['destination']
    if source not in airports or destination not in airports:
        print("One or both airports are not in the dataset.")
    else:
        shortest_paths = dict(nx.all_pairs_shortest_path_length(g_und))
        path = nx.shortest_path(g_und, source=source, target=destination)
        print("The shortest path between", source, "and", destination, "is:", path)
        path_coords = []
        for airport in path:
            SLAT = airports_df.loc[airports_df['IATA'] == airport, 'Latitude'].values[0]
            SLONG = airports_df.loc[airports_df['IATA'] == airport, 'Longitude'].values[0]
            path_coords.append({'latitude': SLAT, 'longitude': SLONG})

        with open('static/path.json', 'w') as f:
            json.dump(path_coords, f)

        return render_template('shortest_path.html', source=source, destination=destination)
    
if __name__ == '__main__':
    app.run(debug=True)