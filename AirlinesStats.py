import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# load the routes and airlines datasets
routes = pd.read_csv('Data/routes.csv', sep=',', header=None)
routes.columns = ['Airline', 'ID', 'SourceAirport', 'SourceAirportID', 'DestAirport', 'DestAirportID',
                  'Codeshare', 'Stops', 'Equipment']
airlines = pd.read_csv('Data/airlines.csv', sep=',', header=None)
airlines.columns = ['ID', 'Name', 'Alias', 'IATA', 'ICAO', 'Callsign', 'Country', 'Active']

# merge the routes and airlines datasets based on the Airline ID
routes = pd.merge(routes, airlines[['ID', 'Name']], on='ID', how='left')

# count the number of routes per airline
top_airlines = routes['Name'].value_counts().nlargest(6).index.tolist()

# filter the routes dataframe to only include the top 6 airlines
routes = routes[routes['Name'].isin(top_airlines)]

# create a map
fig = plt.figure(figsize=(16, 12))
m = Basemap(projection='merc', resolution='l', llcrnrlat=-60, urcrnrlat=80,
            llcrnrlon=-180, urcrnrlon=180)

# draw coastlines, countries, and states
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# plot the routes for each airline in a different color
colors = ['#2ecc71', '#3498db', '#f39c12', '#27ae60', '#85c1e9', '#1abc9c']
for i, airline in enumerate(top_airlines):
    color = colors[i]
    airline_routes = routes[routes['Name'] == airline]
    source_lats = []
    source_lons = []
    dest_lats = []
    dest_lons = []
    for _, row in airline_routes.iterrows():
        source_lat = row['SourceAirportLat']
        source_lon = row['SourceAirportLon']
        dest_lat = row['DestAirportLat']
        dest_lon = row['DestAirportLon']
        source_lats.append(source_lat)
        source_lons.append(source_lon)
        dest_lats.append(dest_lat)
        dest_lons.append(dest_lon)
        m.drawgreatcircle(source_lon, source_lat, dest_lon, dest_lat,
                          linewidth=1, color=color)
    m.scatter(source_lons, source_lats, 5, marker='o', color=color, alpha=0.7)
    m.scatter(dest_lons, dest_lats, 5, marker='o', color=color, alpha=0.7)

# add a title
plt.title('Routes for the Top 6 Airlines', fontsize=20)

# show the map
plt.show()
