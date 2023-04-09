import pandas as pd

# read the routes data from the csv file
routes = pd.read_csv('Data/routes.csv', sep=',', header=None)

# filter the routes with JFK as the source airport
jfk_routes = routes[routes[2] == 'ACA']

# write the resulting routes to a new csv file
jfk_routes.to_csv('jfk_routes.csv', index=False, header=False)
