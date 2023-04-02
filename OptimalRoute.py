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
    SLAT = (airports_df.loc[airports_df['IATA'] == source_airport, 'Latitude'].values[0])
    SLONG = (airports_df.loc[airports_df['IATA'] == source_airport, 'Longitude'].values[0])

    DLAT = (airports_df.loc[airports_df['IATA'] == dest_airport, 'Latitude'].values[0])
    DLONG = (airports_df.loc[airports_df['IATA'] == dest_airport, 'Longitude'].values[0])

    for index, row in routes_df[routes_df['Source airport'] == source_airport].iterrows():
        print(row['Destination airport'])


    distance = haversine(SLAT,SLONG,DLAT,DLONG)



# Load the data
airports_df = pd.read_csv('airports.dat', header=None, names=['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'], index_col=0)
routes_df = pd.read_csv('routes.dat', header=None, names=['Airline', 'Airline ID', 'Source airport', 'Source airport ID', 'Destination airport', 'Destination airport ID', 'Codeshare', 'Stops', 'Equipment'], index_col=0)

# Get user input for the source and destination airports
source = input("Enter the source airport: ")
destination = input("Enter the destination airport: ")


# Run Dijkstra's algorithm to find the shortest path
dijkstra(source, destination)


