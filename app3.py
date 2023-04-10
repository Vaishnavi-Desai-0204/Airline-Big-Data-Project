from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)

# Load the data into a pandas dataframe
routes_df = pd.read_csv('Data/routes.csv', sep=',', header=None)
airports_df = pd.read_csv('Data/airports.csv')

# Rename columns
routes_df.rename(columns={2: 'Source Airport', 4: 'Destination Airport'}, inplace=True)

# Group the routes by origin and destination airports to get the number of flights for each route
route_counts = routes_df.groupby(['Source Airport', 'Destination Airport']).size().reset_index(name='Num flights')

# Sort the routes by the number of flights in descending order
route_counts = route_counts.sort_values(by='Num flights', ascending=False)

# Extract the top 10 busiest routes
top_routes = route_counts.head(10)

# Create a bar chart of the top 10 busiest routes using Plotly
fig = go.Figure()
fig.add_trace(go.Bar(
    x=top_routes['Source Airport'] + ' to ' + top_routes['Destination Airport'],
    y=top_routes['Num flights'],
    name='Top 10 Busiest Routes'
))
fig.update_layout(
    title='Top 10 Busiest Routes',
    xaxis_title='Route',
    yaxis_title='Number of flights',
    xaxis_tickangle=-45
)
plot_div = pyo.offline.plot(fig, output_type='div')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/top_routes')
def top_routes():
    return render_template('top_routes.html', plot_div=plot_div)

if __name__ == '__main__':
    app.run(debug=True)

