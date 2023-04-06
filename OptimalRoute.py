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

import heapq

def dijkstra(source_airport, dest_airport):
    # Initialize distance and visited dictionaries
    distances = {source_airport: 0}
    visited = {}

    # Initialize priority queue with source airport
    pq = [(0, source_airport)]

    while pq:
        # Get the closest airport from the priority queue
        (dist, curr_airport) = heapq.heappop(pq)

        # Check if we've found the destination airport
        if curr_airport == dest_airport:
            return dist

        # Mark the current airport as visited
        visited[curr_airport] = True

        # Update the distances of neighboring airports
        for index, row in routes_df[routes_df['Source airport'] == curr_airport].iterrows():
            neighbor_airport = row['Destination airport']
            neighbor_distance = row['Distance']
            if neighbor_airport not in visited:
                new_distance = dist + neighbor_distance
                if neighbor_airport not in distances or new_distance < distances[neighbor_airport]:
                    distances[neighbor_airport] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor_airport))

    # If we get here, there is no path from source to dest
    return float('inf')


# Load the data
airports_df = pd.read_csv('airports.dat', header=None, names=['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'], index_col=0)
routes_df = pd.read_csv('routes.dat', header=None, names=['Airline', 'Airline ID', 'Source airport', 'Source airport ID', 'Destination airport', 'Destination airport ID', 'Codeshare', 'Stops', 'Equipment'], index_col=0)

# Get user input for the source and destination airports
source = input("Enter the source airport: ")
destination = input("Enter the destination airport: ")


# Run Dijkstra's algorithm to find the shortest path
dijkstra(source, destination)


