import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import collections
from IPython.display import Image

routes = pd.read_csv('Data/routes.csv',sep=',',header=None)
routes = routes.drop([0,1,3,5,6,7,8],axis=1)
routes.rename(columns={2: 'origin',4: 'destination'}, inplace=True)

isolated = ['BMY', 'GEA', 'ILP', 'KNQ', 'KOC', 'LIF', 'MEE', 'TGJ', 'TOU', 'UVE','ERS', 'MPA', 'NDU', 'OND','BFI', 'CLM', 'ESD', 'FRD','AKB', 'DUT', 'IKO', 'KQA','SPB', 'SSB','CKX', 'TKJ','BLD', 'GCW']


routes = routes[np.logical_not(routes.origin.isin(isolated))]
routes = routes[np.logical_not(routes.destination.isin(isolated))]

llista1=np.unique(routes.origin)
llista2=np.unique(routes.destination)
airports = np.unique(np.concatenate([llista1,llista2],axis=0))

routes_with_weight = routes.groupby(['origin', 'destination']).size()

links = routes_with_weight.index.unique()
weights = routes_with_weight.tolist()

input_tuple = [link + (weight,) for link, weight in zip (links, weights)]

g_und = nx.Graph()
g_und.name = 'Undirected Graph'
g_und.add_nodes_from(airports,bipartite=1)
g_und.add_weighted_edges_from(input_tuple)

degree = g_und.degree()
df = pd.DataFrame()
df['val'] = dict(degree).values()
df['key'] = dict(degree).keys()


cluster_coeff=nx.clustering(g_und)

shortest_paths = dict(nx.all_pairs_shortest_path_length(g_und))

l = shortest_paths
k=6

def separation_degrees(shortest_paths, with_n_jumps):
    n = len(shortest_paths)
    coef = [None] * n
    for shortest_path, i in zip(shortest_paths.values(), range(n)):
        jumps = np.array(list(shortest_path.values()))
        # Ratio of airports that are reachable with n jumps
        coef[i] = float(len(jumps[jumps <= with_n_jumps])) / float(n-1)
    return round(np.mean(coef), 4)


sample = []
for i in np.arange(1,10):
    perc = separation_degrees(shortest_paths,i)
    sample.append(perc)

def jumps_to_reach_any_airport(shortest_paths,city):
    max_jump = [{i: max(shortest_paths[i].values())} for i in shortest_paths]
    finalMap = {}
    for max_jump_per_cities in max_jump:
        finalMap.update(max_jump_per_cities)
    return finalMap[city]

newL = [{i: max(shortest_paths[i].values())} for i in shortest_paths]
finalMap = {}
for d in newL:
    finalMap.update(d)

world_hubs = [{i:j} for i,j in zip(finalMap,finalMap.values()) if j==min(finalMap.values())]

from mpl_toolkits.basemap import Basemap

airports_df = pd.read_csv('Data/airports.dat', header=None, names=['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'])
airports_df = airports_df.set_index('IATA')

# select only the columns we need
airports = airports_df[['Latitude', 'Longitude']]
hubs = [list(d.keys())[0] for d in world_hubs]

hubs_df = airports.loc[hubs]

# create a Basemap instance centered on the world
m = Basemap(projection='mill', lon_0=0)

# draw coastlines and country borders
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)

# iterate through the hubs and add a marker for each one
for i, hub in hubs_df.iterrows():
    # get the coordinates of the hub
    lat, lon = hub['Latitude'], hub['Longitude']
    # convert the coordinates to the map projection
    x, y = m(lon, lat)
    # add a marker to the map
    m.plot(x, y, marker='o', markersize=10, markerfacecolor='red')
    plt.text(x + 10000, y + 10000, i, fontsize=8, fontweight='bold')

# show the map
plt.show()
