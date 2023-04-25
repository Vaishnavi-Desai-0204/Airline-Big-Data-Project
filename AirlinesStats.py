import pandas as pd
import matplotlib.pyplot as plt

# load the routes and airlines datasets
routes = pd.read_csv('Data/routes.csv', sep=',', header=None)
routes.columns = ['Airline', 'ID', 'SourceAirport', 'SourceAirportID', 'DestAirport', 'DestAirportID',
                  'Codeshare', 'Stops', 'Equipment']
airlines = pd.read_csv('Data/airlines.csv', sep=',', header=None)
airlines.columns = ['ID', 'Name', 'Alias', 'IATA', 'ICAO', 'Callsign', 'Country', 'Active']

# merge the routes and airlines datasets based on the Airline ID
routes = pd.merge(routes, airlines[['ID', 'Name']], on='ID', how='left')

# count the number of routes per airline
routes_by_airline = routes['Name'].value_counts().nlargest(6)
# create a pie chart with custom colors
colors = ['#2ecc71', '#3498db', '#f39c12', '#27ae60', '#85c1e9', '#1abc9c']
fig, ax = plt.subplots()
ax.pie(routes_by_airline, labels=routes_by_airline.index, autopct='%1.1f%%', colors=colors)
ax.set_title('Top 6 Airlines by Number of Routes')

# add a legend with the corresponding colors
legend_labels = ['{}\n{}'.format(name, count) for name, count in zip(routes_by_airline.index, routes_by_airline.values)]
ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.show()
