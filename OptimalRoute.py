import pandas as pd
import plotly.express as px
import math
import heapq

# Define a function to calculate the distance between two airports using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Radius of the earth in km
    dLat = math.radians(lat2-lat1)
    dLon = math.radians(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c # Distance in km
    return d

def dijkstra(source_airport, dest_airport):
    
    distances = {}
    queue = []
    
    previous = []
    current_airport = source_airport
    visited = {}

    for index, row in routes_df.iterrows():
        dest_airport = row['Destination airport']
        distances[dest_airport] = math.inf

    distances[source_airport] = 0
    queue = [(0, source_airport)]

    while len(queue) > 0:
        for index, row in routes_df[routes_df['Source airport'] == current_airport].iterrows():
            neighbor_airport = row['Destination airport']

            SLAT = (airports_df.loc[airports_df['IATA'] == current_airport, 'Latitude'].values[0])
            SLONG = (airports_df.loc[airports_df['IATA'] == current_airport, 'Longitude'].values[0])
            DLAT = (airports_df.loc[airports_df['IATA'] == neighbor_airport, 'Latitude'].values[0])
            DLONG = (airports_df.loc[airports_df['IATA'] == neighbor_airport, 'Longitude'].values[0])
            distance = haversine(SLAT,SLONG,DLAT,DLONG)

            if(distances[neighbor_airport] > distances[current_airport] + distance):
                distances[neighbor_airport] = distances[current_airport] + distance
                heapq.heappush(queue, (distance, neighbor_airport))

        current_airport = heapq.heappop(queue)
        visited[current_airport] = True
   

# Load the data
airports_df = pd.read_csv('Data/airports.dat', header=None, names=['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'], index_col=0)
routes_df = pd.read_csv('Data/routes.dat', header=None, names=['Airline', 'Airline ID', 'Source airport', 'Source airport ID', 'Destination airport', 'Destination airport ID', 'Codeshare', 'Stops', 'Equipment'], index_col=0)

# Get user input for the source and destination airports
source = input("Enter the source airport: ")
destination = input("Enter the destination airport: ")


# Run Dijkstra's algorithm to find the shortest path
dijkstra(source, destination)


