import pandas as pd
import numpy as np
import networkx as nx
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

routes = pd.read_csv('Data/routes.csv',sep=',',header=None)
routes = routes.drop([0,1,3,5,6,7,8],axis=1)
routes.rename(columns={2: 'origin',4: 'destination'}, inplace=True)

isolated = ['BMY', 'GEA', 'ILP', 'KNQ', 'KOC', 'LIF', 'MEE', 'TGJ', 'TOU', 'UVE','ERS', 'MPA', 'NDU', 'OND','BFI', 'CLM', 'ESD', 'FRD','AKB', 'DUT', 'IKO', 'KQA','SPB', 'SSB','CKX', 'TKJ','BLD', 'GCW']

airports_df = pd.read_csv('Data/airports.dat', header=None, names=['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'], index_col=0)

routes = routes[np.logical_not(routes.origin.isin(isolated))]
routes = routes[np.logical_not(routes.destination.isin(isolated))]

llista1 = np.unique(routes.origin)
llista2 = np.unique(routes.destination)
airports = np.unique(np.concatenate([llista1,llista2],axis=0))

routes_with_weight = routes.groupby(['origin', 'destination']).size()
links = routes_with_weight.index.unique()
weights = routes_with_weight.tolist()

input_tuple = [link + (weight,) for link, weight in zip (links, weights)]

g_und = nx.Graph()
g_und.name = 'Undirected Graph'
g_und.add_nodes_from(airports,bipartite=1)
g_und.add_weighted_edges_from(input_tuple)


source = input("Enter the source airport code: ").upper()
destination = input("Enter the destination airport code: ").upper()

if source not in airports or destination not in airports:
    print("One or both airports are not in the dataset.")
else:
    shortest_paths = dict(nx.all_pairs_shortest_path_length(g_und))
    path = nx.shortest_path(g_und, source=source, target=destination)
    print("The shortest path between", source, "and", destination, "is:", path)


fig = plt.figure(figsize=(12, 8))
m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=85, llcrnrlon=-180, urcrnrlon=180, resolution='l')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='white', alpha=0.3)

for airport in path:
    SLAT = (airports_df.loc[airports_df['IATA'] == airport, 'Latitude'].values[0])
    SLONG = (airports_df.loc[airports_df['IATA'] == airport, 'Longitude'].values[0])
    x, y = m(SLONG , SLAT)
    m.plot(x, y, marker='o', markersize=10, markerfacecolor='red', alpha=0.8)

for i in range(len(path)-1):
    start_airport = path[i]
    end_airport = path[i+1]
    start_lat = airports_df.loc[airports_df['IATA'] == start_airport, 'Latitude'].values[0]
    start_lon = airports_df.loc[airports_df['IATA'] == start_airport, 'Longitude'].values[0]
    end_lat = airports_df.loc[airports_df['IATA'] == end_airport, 'Latitude'].values[0]
    end_lon = airports_df.loc[airports_df['IATA'] == end_airport, 'Longitude'].values[0]
    x_start, y_start = m(start_lon, start_lat)
    x_end, y_end = m(end_lon, end_lat)
    m.plot([x_start, x_end], [y_start, y_end], linewidth=2, color='blue')
    m.plot(x_start, y_start, marker='o', markersize=10, markerfacecolor='red', alpha=0.8)
    m.plot(x_end, y_end, marker='o', markersize=10, markerfacecolor='red', alpha=0.8)

    

plt.show()