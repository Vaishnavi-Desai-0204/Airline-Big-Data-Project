import pandas as pd
import plotly.express as px

# Load the data
airports_df = pd.read_csv('airports.dat', header=None, names=['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source'], index_col=0)

# Plot the data
fig = px.scatter_mapbox(airports_df, lat="Latitude", lon="Longitude", hover_name="Name", hover_data=["City", "Country", "IATA", "ICAO"],
                        zoom=1, height=500)
fig.update_layout(mapbox_style="stamen-terrain")
fig.show()
