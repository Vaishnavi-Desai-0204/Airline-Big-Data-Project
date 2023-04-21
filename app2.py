import pandas as pd
from sklearn.cluster import KMeans

# Load the OpenFlights data
routes_df = pd.read_csv('Data/routes.csv')
print(routes_df.columns)
# Preprocess the data
airline_freq = routes_df['airline'].value_counts()
airline_freq_df = pd.DataFrame({'airline': airline_freq.index, 'frequency': airline_freq.values})

# Extract the source and destination airports
source_airports = routes_df['source airport'].unique()
destination_airports = routes_df['destination airport'].unique()
airports = set(source_airports).union(set(destination_airports))

# Create a mapping from airports to indices
airport_to_index = {airport: i for i, airport in enumerate(airports)}

# Create a matrix of route frequencies
route_matrix = pd.DataFrame(0, index=airline_freq_df['airline'], columns=range(len(airports)))
for index, row in routes_df.iterrows():
    airline = row['airline']
    source_airport = row['source airport']
    destination_airport = row['destination airport']
    route_matrix.loc[airline, airport_to_index[source_airport]] += 1
    route_matrix.loc[airline, airport_to_index[destination_airport]] += 1

# Cluster the data
kmeans = KMeans(n_clusters=5).fit(route_matrix)

# Visualize the results
import matplotlib.pyplot as plt
plt.scatter(route_matrix.values[:, 0], route_matrix.values[:, 1], c=kmeans.labels_)
plt.xlabel('Source Airport')
plt.ylabel('Destination Airport')
plt.show()
