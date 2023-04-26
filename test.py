import pandas as pd
import json

# load the airports dataset
airports = pd.read_csv('Data/airports.csv', sep=',', header=None)
airports.columns = ['ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude',
                    'Altitude', 'Timezone', 'DST', 'TzDatabaseTimezone', 'Type', 'Source']

# take user input for a country
country = input('Enter a country name: ')

# filter the airports dataframe to show all airports in the user-specified country
country_airports = airports[airports['Country'] == country]
print(country_airports['Latitude'])
# print the airports
print(country_airports['Name'])

# loop through each airport in the country and find its cluster
for airport_name in country_airports['Name']:
    # initialize the cluster number to -1 (meaning not found)
    cluster_number = -1
    
    # loop through the cluster files to find the airport
    for cluster in range(6):
        with open(f'static/cluster_{cluster}.json', 'r') as f:
            cluster_data = json.load(f)
            
            # loop through the airports in the cluster
            for airport in cluster_data:
                # check if the airport name matches
                if airport[4] == airport_name:
                    # set the cluster number and break out of the loop
                    cluster_number = cluster
                    break
            
            # break out of the loop if the cluster number is found
            if cluster_number != -1:
                break
    
    # print the result
    if cluster_number != -1:
        print(f"The airport {airport_name} in {country} belongs to cluster {cluster_number}.")
    else:
        print(f"The airport {airport_name} in {country} was not found in any cluster.")


