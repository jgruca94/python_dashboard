import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os


df = pd.read_csv('data/full_data.csv')

def get_competition_data(folder_path='Results/COMPETITION/'):
    # List to store DataFrames
    dfs = []

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # Read CSV file and append DataFrame to list
            df = pd.read_csv(file_path, sep=';')
            dfs.append(df)

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df

def load_data(date, product):
    # Load data from CSV file based on selected date and product
    filename = f"Decisions/decision_{date}.csv"  # Assuming filename convention follows this format
    df = pd.read_csv(filename)
    if product is not None:
        df = df[df['PRODID'] == product]
    return df

subdf = df[['IMPRESSIONS', 'WEEKDAY', 'HOUR', 'PRODID', 'CR', 'CTR']]
grouped = subdf.groupby(['WEEKDAY', 'HOUR', 'PRODID'])
hourly_averages = grouped.agg({
    'IMPRESSIONS': 'mean',
    'CR': 'mean',
    'CTR': 'mean'
}).reset_index()


lstm_df_prod1_week1 = pd.read_csv('data/LSTM/lstm_prod_1_week1.csv')
lstm_df_prod1_week2 = pd.read_csv('data/LSTM/lstm_prod_1_week2.csv')

forecast_concatenated_df = pd.concat([lstm_df_prod1_week1, lstm_df_prod1_week2])
forecast_unique_dates = sorted(forecast_concatenated_df['DATE'].unique())


# Define the start and end dates
start_date = datetime.strptime("2024-01-26", "%Y-%m-%d")
end_date = datetime.strptime("2024-02-08", "%Y-%m-%d")

# Initialize the dictionary
weekday_to_dates = {}

# Loop through the dates and populate the dictionary
current_date = start_date
while current_date <= end_date:
    weekday = current_date.strftime("%A")  # Get the weekday name
    if weekday not in weekday_to_dates:
        weekday_to_dates[weekday] = []  # Initialize list if not already present
    weekday_to_dates[weekday].append(current_date.strftime("%Y-%m-%d"))  # Add date to the list
    current_date += timedelta(days=1)  # Move to the next date

def get_weekday(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # Get the weekday name
    weekday = date_obj.strftime("%A")
    
    return weekday

days_forecast = [{'label': f"{date} ({get_weekday(date)})", 
                  'value': forecast_concatenated_df[forecast_concatenated_df['DATE'] == date]['DATE'].iloc[0]} 
                 for date in forecast_unique_dates]



# Filter data based on the first value in the "DATE" column
lstm_init_first_date = lstm_df_prod1_week1['DATE'].iloc[0]
lstm_init_filtered_df = lstm_df_prod1_week1[lstm_df_prod1_week1['DATE'] == lstm_init_first_date]

mean_impressions_df = pd.read_csv('data/MEANS/means.csv')
mean_impressions_filtered_df = mean_impressions_df[mean_impressions_df['WEEKDAY'] == 'Friday']
mean_impressions_filtered_df = mean_impressions_filtered_df[mean_impressions_filtered_df['PRODID'] == 1]

competition_df = get_competition_data()

competition_days = competition_df['DATE'].unique()
competition_days = [{'label': f"{date}", 'value': date}
        for date in competition_days]
competition_days_sorted = sorted(competition_days, key=lambda x: x['value'], reverse=True)
competition_df = competition_df[competition_df['PRODID'] == 1]
competition_df = competition_df[competition_df['DATE'] == "2023-12-12"]
spot1 = competition_df[competition_df['POSITION'] == 1]
spot2 = competition_df[competition_df['POSITION'] == 2]
spot3 = competition_df[competition_df['POSITION'] == 3]
spot4 = competition_df[competition_df['POSITION'] == 4]


initial_competition_graph = go.Figure()

initial_competition_graph.add_trace(go.Line(x=spot1['HOUR'],
                                           y=spot1['BID'],
                                           name='Spot 1',
                                           marker_color='green'))

initial_competition_graph.add_trace(go.Scatter(x=spot2['HOUR'],
                                               y=spot2['BID'],
                                               mode='lines',
                                               name='Spot 2',
                                               line=dict(color='blue')))

initial_competition_graph.add_trace(go.Scatter(x=spot3['HOUR'],
                                               y=spot3['BID'],
                                               mode='lines',
                                               name='Spot 3',
                                               line=dict(color='red')))

initial_competition_graph.add_trace(go.Scatter(x=spot4['HOUR'],
                                               y=spot4['BID'],
                                               mode='lines',
                                               name='Spot 4',
                                               line=dict(color='orange')))

initial_competition_graph.update_layout(title='Cost for Positions',
                                       xaxis_title='Hours',
                                       yaxis_title='Bids',
                                       legend=dict(x=0, y=1, traceorder='normal'),
                                       xaxis=dict(
                                           tickmode='linear',
                                           dtick=1  # Set the tick size to 1 hour
                                       ))
initial_competition_graph.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )
initial_competition_graph.update_traces(line=dict(color='#F2E3D5')) 


# grp 3
df = df[df['ADVID'] == 3]

piechart_df = df.copy()
piechart_df['DATE'] = pd.to_datetime(piechart_df['DATE'])
piechart_df = piechart_df[piechart_df['DATE'] >= '2024-01-26']
piechart_revenue_df = piechart_df.groupby('PRODID')['REVENUE'].sum().reset_index()
piechart_revenue_labels = piechart_revenue_df['PRODID'].astype(str)

# Create Pie chart for Revenue
initial_piechart_revenue = go.Figure(go.Pie(labels=piechart_revenue_labels, values=piechart_revenue_df['REVENUE'], name="Revenue"))
initial_piechart_revenue.update_traces(hole=.4, hoverinfo="label+value+name")
initial_piechart_revenue.update_layout(
    title_text="Revenue by Product ID",
    annotations=[],
    plot_bgcolor='#012E40',  # Set background color to dark blue
    paper_bgcolor='#012E40', # Set background color to dark blue
    font=dict(color='#F2E3D5') # Set font color to light beige
)
piechart_cost_df = piechart_df.groupby('PRODID')['COST'].sum().reset_index()
piechart_cost_labels = piechart_cost_df['PRODID'].astype(str)
initial_piechart_cost = go.Figure(go.Pie(labels=piechart_cost_labels, values=piechart_cost_df['COST'], name="Cost"))
initial_piechart_cost.update_traces(hole=.4, hoverinfo="label+value+name")
initial_piechart_cost.update_layout(
    title_text="Cost by Product ID",
    annotations=[],
    plot_bgcolor='#012E40',  # Set background color to dark blue
    paper_bgcolor='#012E40', # Set background color to dark blue
    font=dict(color='#F2E3D5') # Set font color to light beige
)

min_date = df["DATE"].min()

# Filter DataFrame for the minimum date
init_df_finance = df[df["DATE"] == '2024-01-26']
init_df_finance = init_df_finance[init_df_finance['PRODID'] == 1]
init_df = df[df["DATE"] == min_date]
init_df = init_df[init_df['PRODID'] == 1]

# labels for dropdown
unique_dates = sorted(df['DATE'].unique(), reverse=True)

date_to_weekday = dict(zip(df['DATE'], df['WEEKDAY']))

days = [{'label': f"{date} ({date_to_weekday[date]})", 'value': df[df['DATE'] == date]['DAY'].iloc[0]} 
        for date in unique_dates]

decision_days_1 = [f"2024-01-{i}" for i in range(26, 32)]
decision_days_2 = [f"2024-02-0{i}" for i in range(1, 10)]
decision_days = decision_days_1 + decision_days_2

decision_days = [{'label': f'{date}', 'value': date} for date in decision_days]

weekdays = subdf['WEEKDAY'].unique()
weekdays = list(weekdays[-1:]) + list(weekdays[:-1])
weekday_labels = [{'label': weekday, 'value': weekday} for weekday in weekdays]

metrics = [
    {'label': 'Impressions', 'value': 'IMPRESSIONS'},
    {'label': 'Conversion Rate', 'value': 'CR'},
    {'label': 'Click-Through-Rate', 'value': 'CTR'},
    {'label': 'Position', 'value': 'POSITION'},
]

metrics2 = [
    {'label': 'Average Impressions', 'value': 'IMPRESSIONS'},
    {'label': 'Average Conversion Rate', 'value': 'CR'},
    {'label': 'Average Click-Through-Rate', 'value': 'CTR'},
]

products = [
    {'label': 'Product 1', 'value': 1},
    {'label': 'Product 2', 'value': 2},
    {'label': 'Product 3', 'value': 3},
        ]

# initial figures
initial_figure1 = px.line(init_df, x='HOUR', y='IMPRESSIONS', title='Impressions by Hour',
                               labels={'IMPRESSIONS': 'IMPRESSIONS'})
initial_figure1.update_xaxes(tickmode='linear', dtick=1)

initial_figure2 = px.line(init_df, x='HOUR', y='POSITION', title='Position by Hour',
                               labels={'POSITION': 'Position'})
initial_figure2.update_xaxes(tickmode='linear', dtick=1)
initial_figure2.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )
initial_figure2.update_traces(line=dict(color='#F2E3D5'))



#Plot boxplot
initial_figure3 = px.box(hourly_averages, x="HOUR", y="IMPRESSIONS",
             title="Average Impressions: Monday",
             labels={"variable": "IMPRESSIONS", "value": "IMPRESSIONS", "WEEKDAY": "Monday"},
             )
initial_figure3.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )
initial_figure3.update_traces(line=dict(color='#F2E3D5'))



# Create the line plot
initial_figure4 = go.Figure()

# Add IMPRESSIONS line
initial_figure4.add_trace(go.Scatter(x=lstm_init_filtered_df['HOUR'],
                         y=lstm_init_filtered_df['IMPRESSIONS'],
                         mode='lines',
                         name='Actual Impressions',
                         line=dict(color='blue')))

# Add PREDICTED_IMPRESSIONS (LSTM) line
initial_figure4.add_trace(go.Scatter(x=lstm_init_filtered_df['HOUR'],
                         y=lstm_init_filtered_df['PREDICTED_IMPRESSIONS'],
                         mode='lines',
                         name='LSTM',))

# Add PREDICTED_IMPRESSIONS (means) line
initial_figure4.add_trace(go.Scatter(x=mean_impressions_filtered_df['HOUR'],
                         y=mean_impressions_filtered_df['PREDICTED_IMPRESSIONS'],
                         mode='lines',
                         name='Daily Averages for Fridays',))

# Update layout
initial_figure4.update_layout(title='Impressions vs. Prediction',
                  xaxis_title='HOURS',
                  yaxis_title='IMPRESSIONS',
                  legend=dict(x=0, y=1, traceorder='normal'),
                  xaxis=dict(
                    tickmode='linear',
                    dtick=1  # Set the tick size to 1 hour
                ))
initial_figure4.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )

initial_figure5 = go.Figure()

# Add REVENUE bar chart
initial_figure5.add_trace(go.Bar(x=init_df_finance['HOUR'],
                                 y=init_df_finance['REVENUE'],
                                 name='Revenue',
                                 marker_color='yellow'))

# Add COST bar chart
initial_figure5.add_trace(go.Bar(x=init_df_finance['HOUR'],
                                 y=init_df_finance['COST'],
                                 name='Cost',
                                 marker_color='red'))

# Add PROFIT line chart
initial_figure5.add_trace(go.Scatter(x=init_df_finance['HOUR'],
                                     y=init_df_finance['REVENUE'] - init_df_finance['COST'],
                                     mode='lines',
                                     name='Profit',
                                     line=dict(color='green', width=2)))  # Specify line properties

# Update layout
initial_figure5.update_layout(title='Financial Report for 2024-01-26',
                              xaxis_title='Hours',
                              yaxis_title='Amount',
                              barmode='group',  # Group bars
                              legend=dict(x=0, y=1, traceorder='normal'),
                              xaxis=dict(
                                  tickmode='linear',
                                  dtick=1  # Set the tick size to 1 hour
                              ))
initial_figure5.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )




#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, dev_tools_ui=False, dev_tools_props_check=False)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

CONTAINER_STYLE = {
    "background-color": "#012E40",  # Change this to the desired background color
}

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 5,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#012E40",
    "color": "#F2E3D5",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "1rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
    "background-color": "#012E40",
    "color": "#F2E3D5",
}

sidebar = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H2("Navigation", className="display-7"),
                html.Hr(),
                dbc.Nav(
                    [
                        dbc.NavLink("Data Visualisation", href="/data", active="exact", style={'color': '#F2E3D5'}),  # Change the color of the active link
                        dbc.NavLink("Financial Summmary", href="/financial_report", active="exact", style={'color': '#F2E3D5'}),
                    ],
                    vertical=True,
                    pills=False,
                ),
            ],
            style={"display": "flex", "flex-direction": "column", "height": "100%"}  # Ensure the card body takes up full height
        ),
    ],
    style={**SIDEBAR_STYLE, "display": "flex", "flex-direction": "column", "height": "100vh"}  # Ensure the sidebar takes up full height
)


content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = dbc.Container([
    dcc.Location(id="url", pathname="/data"),
    dbc.Row([
        dbc.Col(sidebar),
        dbc.Col(content, width=10)
    ])
], fluid=True,
style=CONTAINER_STYLE)


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/" or pathname == "/data":
        return [
                html.H1('Data Visualisation',
                        style={'textAlign':'center', 
                               'color': '#F2E3D5',
                               'padding': '15px'}),
                html.Br(),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='days-dropdown-left',
                            options=days,
                            value=days[10]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                            ),
                        dcc.Dropdown(
                            id='column-dropdown-left',
                            options=metrics,
                            value=metrics[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                        ),
                        dcc.Dropdown(
                            id='product-dropdown-left',
                            options=products,
                            value=products[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',} 
                        ),
                        dcc.Graph(
                            id='tab1_graph1',
                            figure=initial_figure1,
                        ),
                        ]),
                    dbc.Col([
                        dcc.Dropdown(
                            id='days-dropdown-right',
                            options=days,
                            value=days[10]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                            ),
                        dcc.Dropdown(
                            id='column-dropdown-right',
                            options=metrics,
                            value=metrics[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                        ),
                        dcc.Dropdown(
                            id='product-dropdown-right',
                            options=products,
                            value=products[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',} 
                        ),
                        dcc.Graph(
                            id='tab1_graph2',
                            figure=initial_figure2,
                        )
                        ]),
                    ]),
                html.Br(),
                html.Br(),
                html.H1('Per Day Averages',
                        style={'textAlign':'center', 
                               'color': '#F2E3D5',
                               'padding': '15px'}),
                html.Br(),
                dbc.Row([
                        dcc.Dropdown(
                            id='week-dropdown-below',
                            options=weekday_labels,
                            value=weekday_labels[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                            ),
                        dcc.Dropdown(
                            id='column-dropdown-below',
                            options=metrics2,
                            value=metrics[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',} 
                        ),
                        dcc.Dropdown(
                            id='product-dropdown-below',
                            options=products,
                            value=products[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',} 
                        ),
                        dcc.Graph(
                            id='tab1_graph3',
                            figure=initial_figure3,
                        )
                    ]),
                html.Br(),
                html.Br(),
                html.H1('Predicting Impressions',
                        style={'textAlign':'center', 
                               'color': '#F2E3D5',
                               'padding': '15px'}),
                html.Br(),
                dbc.Row([

                        dcc.Dropdown(
                            id='days-dropdown-forecast',
                            options=days_forecast,
                            value=days_forecast[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                            ),
                        dcc.Dropdown(
                            id='product-dropdown-forecast',
                            options=products,
                            value=products[0]['value'],
                            style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',} 
                        ),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='forecast_graph',
                            figure=initial_figure4,
                            ),
                        ]),
                ]),
 
                ]
    elif pathname == "/forecast":
        return [
                html.H1('Forecast',
                        style={'textAlign':'center'}),
                
                ]
    elif pathname == "/financial_report":
        return [
                html.H1('Financial Summary',
                        style={'textAlign':'center'}),
                html.Br(),
                dbc.Row([
                    dcc.Dropdown(
                        id='days-dropdown-performance-control',
                        options=days,
                        value=days[10]['value'],
                        style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                        ),
                    dcc.Dropdown(
                        id='product-dropdown-performance-control',
                        options=products,
                        value=products[0]['value'], 
                        style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                    ),
                ]),
                dbc.Row([
                    dcc.Graph(
                            id='financial-graph',
                            figure=initial_figure5,
                        ),
                ]), 
                html.Br(),
                dbc.Col([
                    dbc.Row([
                        html.Br(),
                        dbc.Col([
                            # html.H2('Budget and Revenue',
                            # style={'textAlign':'center', 
                            #    'color': '#F2E3D5',
                            #    'padding': '15px'}),
                            html.Br(),
                            dbc.Row([
                                dcc.Graph(id='indicator-graph1',
                                        figure={
                                            'data': [
                                                go.Indicator(
                                                    mode = "gauge+number",
                                                    value = df['BUDGET'].min(),
                                                    title = {'text': "Budget left"},
                                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                                    gauge={
                                                        'axis': {'range': [0, 119500]}  # Set the range of the gauge axis
                                                    },
                                                    
                                                )
                                            ],
                                            'layout': 
                                            go.Layout(
                                                height=300,  # Set the height of the graph
                                                width=500,
                                                plot_bgcolor='#012E40',  # Set background color to dark blue
                                                paper_bgcolor='#012E40', # Set background color to dark blue
                                                font=dict(color='#F2E3D5') # Set font color to light beige
                                            )
                                        }),
                                dcc.Graph(id='revenue-piechart',
                                          figure=initial_piechart_revenue,
                                          style={'height': '300px', 'width': '280px', 'margin': '2px auto'}
                                    ),
                                dcc.Graph(id='cost-piechart',
                                          figure=initial_piechart_cost,
                                          style={'height': '300px', 'width': '280px', 'margin': '2px auto'}
                                    ),
                            ]),
                        ]),
                        dbc.Col([
                            html.Br(),
                            dbc.Row([
                                dcc.Dropdown(
                                id='days-dropdown-competition',
                                options=competition_days_sorted,
                                value=competition_days_sorted[0]['value'],
                                style={'color': 'black',
                                   'background-color': '#F2E3D5',
                                   'border-radius': '10px',
                                   'margin': '2px auto',
                                   'height': '35px',
                                   'width': '330px',}
                                ),
                            ]),
                            dbc.Row([
                                dcc.Dropdown(
                                    id='product-dropdown-competition',
                                    options=products,
                                    value=products[0]['value'], 
                                    style={'color': 'black',
                                        'background-color': '#F2E3D5',
                                        'border-radius': '10px',
                                        'margin': '2px auto',
                                        'height': '35px',
                                        'width': '330px',
                                        }
                                ),
                            ]),
                            dbc.Row([
                                dcc.Graph(id='competition-graph',
                                    figure=initial_competition_graph)
                            ]),
                        ])
                    ])   
                ]),               
                ]




# Update tab1_graph1
@app.callback(
    Output('tab1_graph1', 'figure'),
    [Input('days-dropdown-left', 'value'),
    Input('column-dropdown-left', 'value'),
    Input('product-dropdown-left', 'value'),
    ]   
)

def update_tab1_graph1(selected_day, selected_metric, selected_prodid):
    
    filtered_df = df[df['ADVID'] == 3]
    filtered_df = filtered_df[filtered_df['DAY'] == selected_day]
    filtered_df = filtered_df[filtered_df['PRODID'] == selected_prodid]

    if selected_metric == 'IMPRESSIONS':
        y_label = 'IMPRESSIONS'
    elif selected_metric == 'CR':
        y_label = 'CR'
    elif selected_metric == 'CTR':
        y_label = 'CTR'
    elif selected_metric == 'POSITION':
        y_label = 'POSITION'

    date_value = df.loc[df['DAY'] == selected_day, 'DATE'].iloc[0]
    date_value = str(date_value)[:10]

    weekday = df.loc[df['DAY'] == selected_day, 'WEEKDAY'].iloc[0]
    title = f'{weekday}, {date_value}'

    figure = px.line(filtered_df, x='HOUR', y=y_label, title=title,
                           labels={y_label: y_label})
    figure.update_xaxes(tickmode='linear', dtick=1)
    figure.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )
    figure.update_traces(line=dict(color='#F2E3D5')) 

    return figure


@app.callback(
    Output('tab1_graph2', 'figure'),
    [Input('days-dropdown-right', 'value'),
    Input('column-dropdown-right', 'value'),
    Input('product-dropdown-right', 'value'),
    ]   
)

def update_tab1_graph2(selected_day, selected_metric, selected_prodid):
    
    filtered_df = df[df['ADVID'] == 3]
    filtered_df = filtered_df[filtered_df['DAY'] == selected_day]
    filtered_df = filtered_df[filtered_df['PRODID'] == selected_prodid]

    if selected_metric == 'IMPRESSIONS':
        y_label = 'IMPRESSIONS'
    elif selected_metric == 'CR':
        y_label = 'CR'
    elif selected_metric == 'CTR':
        y_label = 'CTR'
    elif selected_metric == 'POSITION':
        y_label = 'POSITION'

    date_value = df.loc[df['DAY'] == selected_day, 'DATE'].iloc[0]
    date_value = str(date_value)[:10]

    weekday = df.loc[df['DAY'] == selected_day, 'WEEKDAY'].iloc[0]
    title = f'{weekday}, {date_value}'

    figure = px.line(filtered_df, x='HOUR', y=y_label, title=title,
                           labels={y_label: y_label})
    figure.update_xaxes(tickmode='linear', dtick=1)
    figure.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )
    figure.update_traces(line=dict(color='#F2E3D5'))

    return figure


# Update tab1_graph3
@app.callback(
    Output('tab1_graph3', 'figure'),
    [Input('week-dropdown-below', 'value'),
    Input('column-dropdown-below', 'value'),
    Input('product-dropdown-below', 'value'),
    ]
)

def update_tab1_graph3(selected_weekday, selected_metric, selected_prodid):
    
    filtered_df = hourly_averages
    filtered_df = filtered_df[filtered_df['WEEKDAY'] == selected_weekday]
    filtered_df = filtered_df[filtered_df['PRODID'] == selected_prodid]

    if selected_metric == 'IMPRESSIONS':
        y_label = 'IMPRESSIONS'
    elif selected_metric == 'CR':
        y_label = 'CR'
    elif selected_metric == 'CTR':
        y_label = 'CTR'
    elif selected_metric == 'POSITION':
        y_label = 'POSITION'


    title = f'Average {selected_metric}: {selected_weekday}'

    figure = px.line(filtered_df, x='HOUR', y=y_label, title=title,
                           labels={y_label: y_label})
    figure.update_xaxes(tickmode='linear', dtick=1)
    figure.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )
    figure.update_traces(line=dict(color='#F2E3D5'))

    return figure


# forecast callback
@app.callback(
    Output('forecast_graph', 'figure'),
    [Input('days-dropdown-forecast', 'value'),
     Input('product-dropdown-forecast', 'value')]
)
def update_forecast(selected_day, selected_product):

    # Define the start and end dates
    start_date = datetime.strptime("2024-01-26", "%Y-%m-%d")
    end_date = datetime.strptime("2024-02-08", "%Y-%m-%d")

    # Initialize the dictionary
    weekday_to_dates = {}

    # Loop through the dates and populate the dictionary
    current_date = start_date
    while current_date <= end_date:
        weekday = current_date.strftime("%A")  # Get the weekday name
        if weekday not in weekday_to_dates:
            weekday_to_dates[weekday] = []  # Initialize list if not already present
        weekday_to_dates[weekday].append(current_date.strftime("%Y-%m-%d"))  # Add date to the list
        current_date += timedelta(days=1)  # Move to the next date

    def get_weekday(date_str):
        # Convert the date string to a datetime object
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # Get the weekday name
        weekday = date_obj.strftime("%A")
        
        return weekday

    weekday = get_weekday(selected_day)

    lstm_df_prod1_week1 = pd.read_csv('data/LSTM/lstm_prod_1_week1.csv')
    lstm_df_prod2_week1 = pd.read_csv('data/LSTM/lstm_prod_2_week1.csv')
    lstm_df_prod3_week1 = pd.read_csv('data/LSTM/lstm_prod_3_week1.csv')

    lstm_df_prod1_week2 = pd.read_csv('data/LSTM/lstm_prod_1_week2.csv')
    lstm_df_prod2_week2 = pd.read_csv('data/LSTM/lstm_prod_2_week2.csv')
    lstm_df_prod3_week2 = pd.read_csv('data/LSTM/lstm_prod_3_week2.csv')

    week1_data = pd.concat([lstm_df_prod1_week1, lstm_df_prod2_week1, lstm_df_prod3_week1])
    week2_data = pd.concat([lstm_df_prod1_week2, lstm_df_prod2_week2, lstm_df_prod3_week2])
    full_data = pd.concat([week1_data, week2_data])

    filtered_df = full_data[full_data['DATE'] == selected_day]
    filtered_df = filtered_df[filtered_df['PRODID'] == selected_product]

    actual_impressions_data = pd.read_csv('data/full_data.csv')
    actual_impressions_data = actual_impressions_data[actual_impressions_data['DATE'] == selected_day]
    actual_impressions_data = actual_impressions_data[actual_impressions_data['PRODID'] == selected_product]

    mean_impressions_df = pd.read_csv('data/MEANS/means.csv')
    mean_impressions_filtered_df = mean_impressions_df[mean_impressions_df['WEEKDAY'] == weekday]
    mean_impressions_filtered_df = mean_impressions_filtered_df[mean_impressions_filtered_df['PRODID'] == selected_product]

    figure = go.Figure()

    # Add IMPRESSIONS line
    figure.add_trace(go.Scatter(x=actual_impressions_data['HOUR'],
                            y=actual_impressions_data['IMPRESSIONS'],
                            mode='lines',
                            name='Actual Impressions',
                            line=dict(color='blue')))

    # Add PREDICTED_IMPRESSIONS (LSTM) line
    figure.add_trace(go.Scatter(x=filtered_df['HOUR'],
                            y=filtered_df['PREDICTED_IMPRESSIONS'],
                            mode='lines',
                            name='LSTM',
                            line=dict(color='orange')))

    # Add PREDICTED_IMPRESSIONS (means) line
    figure.add_trace(go.Scatter(x=mean_impressions_filtered_df['HOUR'],
                            y=mean_impressions_filtered_df['PREDICTED_IMPRESSIONS'],
                            mode='lines',
                            name=f'Average for {weekday}s',
                            line=dict(color='green')))

    # Update layout
    figure.update_layout(title='Actual Impressions vs Predictions',
                    xaxis_title='HOURS',
                    yaxis_title='IMPRESSIONS',
                    legend=dict(x=0, y=1, traceorder='normal'),
                    xaxis=dict(
                        #tickmode='linear',
                        dtick=1  # Set the tick size to 1 hour
                    ))
    figure.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )

    return figure

@app.callback(
    Output('financial-graph', 'figure'),
    [Input('days-dropdown-performance-control', 'value'),
     Input('product-dropdown-performance-control', 'value')]
)
def update_performance_graph(selected_day, selected_product):

    df = pd.read_csv('data/full_data.csv')
    subdf = df[df['DAY'] == selected_day]
    subdf = subdf[subdf['PRODID'] == selected_product]
    day = subdf['DATE'].iloc[0]

    figure = go.Figure()

    # Add REVENUE bar chart
    figure.add_trace(go.Bar(x=subdf['HOUR'],
                                    y=subdf['REVENUE'],
                                    name='Revenue',
                                    marker_color='yellow'))

    # Add COST bar chart
    figure.add_trace(go.Bar(x=subdf['HOUR'],
                                    y=subdf['COST'],
                                    name='Cost',
                                    marker_color='red'))

    # Add PROFIT line chart
    figure.add_trace(go.Scatter(x=subdf['HOUR'],
                                        y=subdf['REVENUE'] - subdf['COST'],
                                        mode='lines',
                                        name='Profit',
                                        line=dict(color='#5D8679', width=2)))  # Specify line properties

    # Update layout
    figure.update_layout(title=f'Financial Report for {day}',
                                xaxis_title='Hours',
                                yaxis_title='Amount',
                                barmode='group',  # Group bars
                                legend=dict(x=0, y=1, traceorder='normal'),
                                xaxis=dict(
                                    tickmode='linear',
                                    dtick=1  # Set the tick size to 1 hour
                                ))
    figure.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    )

    return figure

@app.callback(
    Output('competition-graph', 'figure'),
    [Input('days-dropdown-competition', 'value'),
     Input('product-dropdown-competition', 'value')]
)
def update_position_cost_graph(selected_date, selected_product):
    
    competition_df = get_competition_data()
    competition_df = competition_df[competition_df['PRODID'] == selected_product]
    competition_df = competition_df[competition_df['DATE'] == selected_date]
    spot1 = competition_df[competition_df['POSITION'] == 1]
    spot2 = competition_df[competition_df['POSITION'] == 2]
    spot3 = competition_df[competition_df['POSITION'] == 3]
    spot4 = competition_df[competition_df['POSITION'] == 4]


    figure = go.Figure()

    figure.add_trace(go.Line(x=spot1['HOUR'],
                                            y=spot1['BID'],
                                            name='Spot 1',
                                            marker_color='green'))

    figure.add_trace(go.Scatter(x=spot2['HOUR'],
                                                y=spot2['BID'],
                                                mode='lines',
                                                name='Spot 2',
                                                line=dict(color='blue')))

    figure.add_trace(go.Scatter(x=spot3['HOUR'],
                                                y=spot3['BID'],
                                                mode='lines',
                                                name='Spot 3',
                                                line=dict(color='red')))

    figure.add_trace(go.Scatter(x=spot4['HOUR'],
                                                y=spot4['BID'],
                                                mode='lines',
                                                name='Spot 4',
                                                line=dict(color='orange')))

    figure.update_layout(title='Cost for Positions',
                                        xaxis_title='Hours',
                                        yaxis_title='Bids',
                                        legend=dict(x=0, y=1, traceorder='normal'),
                                        xaxis=dict(
                                            tickmode='linear',
                                            dtick=1  # Set the tick size to 1 hour
                                        ))
    figure.update_layout(
        plot_bgcolor='#012E40',  # Set background color to dark blue
        paper_bgcolor='#012E40', # Set background color to dark blue
        font=dict(color='#F2E3D5') # Set font color to light beige
    ) 
    

    return figure



if __name__=='__main__':
    app.run_server(debug=True, port=3000)
