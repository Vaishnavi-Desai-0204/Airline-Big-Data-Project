from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import networkx as nx
import json
from flask import Flask, render_template
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)

# Load the data into a pandas dataframe
routes_df = pd.read_csv('Data/routes.csv', sep=',', header=None)
airports_df = pd.read_csv('Data/airports.csv')

# Rename columns
routes_df.rename(columns={2: 'Source Airport', 4: 'Destination Airport'}, inplace=True)

# Group the routes by origin and destination airports to get the number of flights for each route
route_counts = routes_df.groupby(['Source Airport', 'Destination Airport']).size().reset_index(name='Num flights')

# Sort the routes by the number of flights in descending order
route_counts = route_counts.sort_values(by='Num flights', ascending=False)

# Extract the top 10 busiest routes
top_routes = route_counts.head(10)

# Create a bar chart of the top 10 busiest routes using Plotly
fig = go.Figure()
fig.add_trace(go.Bar(
    x=top_routes['Source Airport'] + ' to ' + top_routes['Destination Airport'],
    y=top_routes['Num flights'],
    name='Top 10 Busiest Routes'
))
fig.update_layout(
    title='Top 10 Busiest Routes',
    xaxis_title='Route',
    yaxis_title='Number of flights',
    xaxis_tickangle=-45
)
plot_div = pyo.offline.plot(fig, output_type='div')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/top_routes')
def top_routes_chart():
    return render_template('top_routes.html', plot_div=plot_div)

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
        #shortest_paths = dict(nx.all_pairs_shortest_path_length(g_und))
        path = nx.shortest_path(g_und, source=source, target=destination)
        print("The shortest path between", source, "and", destination, "is:", path)
        arcdata = []
        for i in range(len(path)-1):
            source = path[i]
            target = path[i+1]
            SLAT1 = airports_df.loc[airports_df['IATA'] == source, 'Latitude'].values[0]
            SLONG1 = airports_df.loc[airports_df['IATA'] == source, 'Longitude'].values[0]
            SLAT2 = airports_df.loc[airports_df['IATA'] == target, 'Latitude'].values[0]
            SLONG2 = airports_df.loc[airports_df['IATA'] == target, 'Longitude'].values[0]
            arc = {"sourceLocation": [SLONG1, SLAT1], "targetLocation": [SLONG2, SLAT2]}
            arcdata.append(arc)
        bubbles = []
        for i in range(len(path)):
            airport = path[i]
            lat = airports_df.loc[airports_df['IATA'] == airport, 'Latitude'].values[0]
            long = airports_df.loc[airports_df['IATA'] == airport, 'Longitude'].values[0]
            bubble = [long, lat, 5.0 , airport]
            bubbles.append(bubble)

    with open('static/arcdata.json', 'w') as f:
        json.dump(arcdata, f)

    # Save bubbles data to JSON file
    with open('static/bubbles.json', 'w') as f:
        json.dump(bubbles, f)

    return render_template('shortest_path.html')

if __name__ == '__main__':
    app.run(debug=True)