import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data into a pandas dataframe
routes_df = pd.read_csv('routes.csv')
airports_df = pd.read_csv('airports.csv')

# Group the routes by origin and destination airports to get the number of flights for each route
route_counts = routes_df.groupby(['Source Airport', 'Destination Airport']).size().reset_index(name='Num flights')
print ("route counts = ", route_counts)

# Sort the routes by the number of flights in descending order
route_counts = route_counts.sort_values(by='Num flights', ascending=False)

# Extract the top 10 busiest routes
top_routes = route_counts.head(10)

# Print the top 10 busiest routes
print(top_routes)

# Plot a bar chart of the top 10 busiest routes
plt.bar(x=top_routes['Source Airport'] + ' to ' + top_routes['Destination Airport'], height=top_routes['Num flights'])
plt.xticks(rotation=90)
plt.xlabel('Route')
plt.ylabel('Number of flights')
plt.title('Top 10 busiest routes')
plt.show()

# Use k-means clustering to cluster the airports based on their popularity
airport_counts = pd.concat([routes_df['Source Airport'], routes_df['Destination Airport']], ignore_index=True)
airport_counts = airport_counts.value_counts().reset_index(name='Num flights')
airport_counts = airport_counts.rename(columns={'index': 'Airport'})
print(airport_counts.loc[(airport_counts['Airport'] == 'ORD')])

kmeans = KMeans(n_clusters=6, random_state=0).fit(airport_counts[['Num flights']])


# Print the cluster assignments for each airport
airport_counts['Cluster'] = kmeans.labels_
print(airport_counts)

# Find the cluster with the most number of airports
# max_cluster = airport_counts['Cluster'].value_counts().reset_index(name='Num flights')
# max_cluster = airport_counts.groupby('Cluster')['Num flights'].sum()

# Calculate the average flight count for each cluster
cluster_means = airport_counts.groupby('Cluster')['Num flights'].mean()

# Get the cluster with the highest average flight count
max_cluster = cluster_means.idxmax()

print(max_cluster)

# Print the airports in the cluster with the most number of airports
print(airport_counts[airport_counts['Cluster'] == max_cluster]['Airport'])

# plt.scatter(x=airport_counts['Airport'], y=airport_counts['Num flights'], c=airport_counts['Cluster'])
# plt.xticks(rotation=90)
# plt.xlabel('Airport')
# plt.ylabel('Number of flights')
# plt.title('Airport popularity clusters')
# plt.show()

# Find the ideal altitude to build airports in the future

popular_airports_codes = airport_counts[airport_counts['Cluster'] == max_cluster]['Airport']
lookup_df = airports_df[airports_df['IATA'].isin(popular_airports_codes)]
altitudes = lookup_df['Altitude']
ideal_altitude = altitudes.mean()

# airport_elevations = pd.read_csv('airports.csv', usecols=['Name', 'Altitude'])
# airport_elevations = airport_elevations.dropna()
# airport_elevations = airport_elevations[airport_elevations['Altitude'] >= 0]

# ideal_elevation = airport_elevations['Altitude'].mean()
print('The ideal elevation to build airports is', ideal_altitude, 'meters above sea level.')
