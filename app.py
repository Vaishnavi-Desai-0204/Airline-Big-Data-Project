from flask import Flask, render_template, request
import json
import pandas as pd
import numpy as np
import networkx as nx
import json
from flask import Flask, render_template
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.cluster import KMeans

app = Flask(__name__)

#FLASK APPS ------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/top_route')
def top_route():

    # Load the data into a pandas dataframe
    routes_df = pd.read_csv('Data/routes.csv', sep=',', header=None)
    airports_df = pd.read_csv('Data/airports.csv')

    # drop rows where IATA code is NA
    airports_df = airports_df[airports_df['IATA'] != '\\N']
    # airports_df = airports_df[airports_df['Altitude'] != '\\NaN']

    # Rename columns
    routes_df.rename(columns={2: 'Source Airport', 4: 'Destination Airport'}, inplace=True)

    # Group the routes by origin and destination airports to get the number of flights for each route
    route_counts = routes_df.groupby(['Source Airport', 'Destination Airport']).size().reset_index(name='Num flights')

    # Sort the routes by the number of flights in descending order
    route_counts = route_counts.sort_values(by='Num flights', ascending=False)

    # Extract the top 10 busiest routes
    top_routes = route_counts.head(10)
    top_routes_json = top_routes.to_dict('records')
    with open('static/top_routes_1.json', 'w') as fp:
        json.dump(top_routes_json, fp)

    return render_template('top_route.html')

@app.route('/airport_kmeans')
def airport_kmeans():
    # Load the data into a pandas dataframe
    routes_df = pd.read_csv('Data/routes.csv', sep=',', header=None)
    airports_df = pd.read_csv('Data/airports.csv')

    # drop rows where IATA code is NA
    airports_df = airports_df[airports_df['IATA'] != '\\N']

    # Rename columns
    routes_df.rename(columns={2: 'Source Airport', 4: 'Destination Airport'}, inplace=True)

    # Use k-means clustering to cluster the airports based on their popularity (number of flights departing and arriving at each airport)
    airport_counts = pd.concat([routes_df['Source Airport'], routes_df['Destination Airport']], ignore_index=True)
    airport_counts = airport_counts.value_counts().reset_index(name='Num flights')
    airport_counts = airport_counts.rename(columns={'index': 'Airport'})

    kmeans = KMeans(n_clusters=6, random_state=0).fit(airport_counts[['Num flights']])

    # Print the cluster assignments for each airport
    airport_counts['Cluster'] = kmeans.labels_
    # print(airport_counts)

    # Calculate the average flight count for each cluster
    cluster_means = airport_counts.groupby('Cluster')['Num flights'].mean()

    # Get the cluster with the highest average flight count
    max_cluster = cluster_means.idxmax()

    print(max_cluster)

    # Print the airports in the cluster with the highest average number of flights
    busiest_airports = airport_counts[airport_counts['Cluster'] == max_cluster]['Airport']
    print(type(busiest_airports))
    print("busiest airports=", busiest_airports)
    busiest_airports_dict = busiest_airports.to_dict()
    with open('static/busy_airports.json', 'w') as fp:
        json.dump(busiest_airports_dict, fp)


    # Get the altitude of each airport in airport_counts
    airport_counts['Altitude'] = airport_counts['Airport'].map(airports_df.set_index('IATA')['Altitude'])
    # airport_counts['Latitude'] = airport_counts['Airport'].map(airports_df.set_index('IATA')['Latitude'])
    # airport_counts['Longitude'] = airport_counts['Airport'].map(airports_df.set_index('IATA')['Longitude'])
    # airport_counts = airport_counts.dropna(subset=['Altitude'])
    # print(airport_counts)
    
      # Passing on the clusters to centrality.html
    for cluster in range(6):
        cluster_airports = airport_counts[airport_counts['Cluster'] == cluster]
        print(cluster_airports)
        dict_cluster = cluster_airports.to_dict('records')
        with open(f'static/cluster_{cluster}_kmeans.json', 'w') as fp:
            json.dump(dict_cluster, fp)

    # Create scatter plot of altitude vs. number of flights for each airport in the clusters in busiest_airport.html
    airport_counts_dict = airport_counts.to_dict('records')
    with open('static/airport_counts.json', 'w') as fp:
        json.dump(airport_counts_dict, fp)

    return render_template('airport_kmeans.html')




@app.route('/busy_airports')
def busy_airports():
    # Load the data into a pandas dataframe
    routes_df = pd.read_csv('Data/routes.csv', sep=',', header=None)
    airports_df = pd.read_csv('Data/airports.csv')

    # drop rows where IATA code is NA
    airports_df = airports_df[airports_df['IATA'] != '\\N']

    # Rename columns
    routes_df.rename(columns={2: 'Source Airport', 4: 'Destination Airport'}, inplace=True)

    # Use k-means clustering to cluster the airports based on their popularity (number of flights departing and arriving at each airport)
    airport_counts = pd.concat([routes_df['Source Airport'], routes_df['Destination Airport']], ignore_index=True)
    airport_counts = airport_counts.value_counts().reset_index(name='Num flights')
    airport_counts = airport_counts.rename(columns={'index': 'Airport'})
    # print(airport_counts.loc[(airport_counts['Airport'] == 'ORD')])

    kmeans = KMeans(n_clusters=6, random_state=0).fit(airport_counts[['Num flights']])


    # Print the cluster assignments for each airport
    airport_counts['Cluster'] = kmeans.labels_
    # print(airport_counts)

    # Calculate the average flight count for each cluster
    cluster_means = airport_counts.groupby('Cluster')['Num flights'].mean()

    # Get the cluster with the highest average flight count
    max_cluster = cluster_means.idxmax()

    print(max_cluster)

    # Print the airports in the cluster with the highest average number of flights
    busiest_airports = airport_counts[airport_counts['Cluster'] == max_cluster]['Airport']
    print(type(busiest_airports))
    print("busiest airports=", busiest_airports)
    busiest_airports_dict = busiest_airports.to_dict()
    with open('static/busy_airports.json', 'w') as fp:
        json.dump(busiest_airports_dict, fp)


    # Get the altitude of each airport in airport_counts
    airport_counts['Altitude'] = airport_counts['Airport'].map(airports_df.set_index('IATA')['Altitude'])
    airport_counts['Latitude'] = airport_counts['Airport'].map(airports_df.set_index('IATA')['Latitude'])
    airport_counts['Longitude'] = airport_counts['Airport'].map(airports_df.set_index('IATA')['Longitude'])
    airport_counts['Name'] = airport_counts['Airport'].map(airports_df.set_index('IATA')['Name'])
    airport_counts = airport_counts.dropna(subset=['Altitude'])
    # print(airport_counts)
    
# Create scatter plot of altitude vs. number of flights for each airport in the clusters
    colors = [ "pink" , "orange" , "yellow" , "red" , "blue", "green" ]
    size =  [2,13,9,15,7,11]
    for cluster in range(6):
        cluster_lists = []
        cluster_airports = airport_counts[airport_counts['Cluster'] == cluster]
        cluster_airports.drop(['Num flights','Cluster','Altitude'], axis = 1, inplace=True)
        cluster_airports.drop(columns=['Airport'], inplace=True)
        values = cluster_airports.values.tolist()
        print(values)
        for val in values:
            lat, lon , airport= val[1], val[0] , val[2]
            val = [lat, lon, size[cluster] , colors[cluster], airport]
            cluster_lists.append(val)
        with open(f'static/cluster_{cluster}.json', 'w') as fp:
            json.dump(cluster_lists, fp)


    return render_template('busy_airports.html')

@app.route('/')
def main_page():
    return render_template('main_page.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/centrality')
def centrality():
    return render_template('centrality.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/routes')
def routes():
    return render_template('routes.html')

@app.route('/shortest_path', methods=['POST'])
def shortest_path():
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
        all_shortest_paths = list(nx.all_shortest_paths(g_und, source=source, target=destination))
        top_3_paths = all_shortest_paths[:3]
        bubbles = []
        arcdata = []
        for path in top_3_paths:
            for i in range(len(path)-1):
                source = path[i]
                target = path[i+1]
                weight = g_und[source][target]['weight']
                SLAT1 = airports_df.loc[airports_df['IATA'] == source, 'Latitude'].values[0]
                SLONG1 = airports_df.loc[airports_df['IATA'] == source, 'Longitude'].values[0]
                SLAT2 = airports_df.loc[airports_df['IATA'] == target, 'Latitude'].values[0]
                SLONG2 = airports_df.loc[airports_df['IATA'] == target, 'Longitude'].values[0]
                arc = {"sourceLocation": [SLONG1, SLAT1], "targetLocation": [SLONG2, SLAT2], "weight": weight}
                arcdata.append(arc)
     
            for i in range(len(path)):
                airport = path[i]
                lat = airports_df.loc[airports_df['IATA'] == airport, 'Latitude'].values[0]
                long = airports_df.loc[airports_df['IATA'] == airport, 'Longitude'].values[0]
                name = airports_df.loc[airports_df['IATA'] == airport, 'Name'].values[0]
                bubble = [long, lat, 5.0 , airport , name]
                bubbles.append(bubble)

    with open('static/arcdata.json', 'w') as f:
        json.dump(arcdata, f)

    # Save bubbles data to JSON file
    with open('static/bubbles.json', 'w') as f:
        json.dump(bubbles, f)

    return render_template('shortest_path.html')

if __name__ == '__main__':
    app.run(debug=True)