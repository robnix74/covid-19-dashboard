import kaggle

import pandas as pd
import numpy as np

import os
import datetime as dt

from urllib.request import urlopen
import json

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import plotly.offline as pyo
import plotly.graph_objs as go

#kaggle.api.authenticate()
#kaggle.api.dataset_download_files('sudalairajkumar/covid19-in-india', path='./covid', unzip=True)

for i in list(map(lambda x: x.split('.csv')[0],os.listdir('./covid'))):
    exec(i + ' = pd.read_csv("./covid/' + i +'.csv")')

#------- Standardising Datasets -------#

def get_ICMRTestingDetails_date(param):
    val = param.split()
    return pd.Series([val[0], val[1]])

ICMRTestingDetails[['Date', 'Time']] = ICMRTestingDetails.DateTime.apply(get_ICMRTestingDetails_date)

covid_19_india = covid_19_india.rename(columns={'Cured' : 'Recovered', 'Deaths' : 'Deceased'})

covid_19_india['Active'] = covid_19_india.Confirmed - (covid_19_india.Deceased + covid_19_india.Recovered)

# Standardising Dates

if str(covid_19_india.Date.iloc[0]).find('/') != -1:
    try:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d/%m/%y')
    except:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d/%m/%Y')
else:
    try:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d-%m-%y')
    except:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d-%m-%Y')

if str(ICMRTestingDetails.Date.iloc[0]).find('/') != -1:
    try:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d/%m/%y')
    except:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d/%m/%Y')
else:
    try:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d-%m-%y')
    except:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d-%m-%Y')

# Getting Specific Dates

covid_19_recent_date = sorted(set(covid_19_india.Date))[-1]
icmr_recent_date = sorted(set(ICMRTestingDetails.Date))[-1]

today = dt.date.today().strftime('%d-%m-%y')
yesterday = (dt.date.today() - dt.timedelta(1)).strftime('%d-%m-%y')
day_bef_yest = (dt.date.today() - dt.timedelta(2)).strftime('%d-%m-%y')

daywise = covid_19_india.groupby('Date')

def get_current_stats():
    return [daywise.sum().loc[covid_19_recent_date], int(ICMRTestingDetails[ICMRTestingDetails.Date == icmr_recent_date]['TotalSamplesTested'])]

current_stats = get_current_stats()

latest_recovered, latest_deceased, latest_confirmed, latest_active = current_stats[0][['Recovered', 'Deceased', 'Confirmed', 'Active']].values
latest_tested = current_stats[1]

#------- Map Data -------#

current_stats_statewise = covid_19_india.groupby(['Date', 'State/UnionTerritory']).sum().loc[covid_19_recent_date]

state_json_file = open('indian_states_geojson.json')
indian_states = json.load(state_json_file)

for ele in indian_states['features']:
    ele.update({'id' : str(ele['properties']['ID_1'])})

# Extracting the state and feature id's required to plot the map

state_names_ids = []
for ele in indian_states['features']:
    state_names_ids.append([ele['properties']['NAME_1'], ele['id']])

state_fid = pd.DataFrame(state_names_ids, columns = ['State', 'fid'])
state_fid['State'] = state_fid['State'].replace('Uttaranchal', 'Uttarakhand')
state_fid['State'] = state_fid['State'].replace('Telangana', 'Telengana')
state_fid['State'] = state_fid['State'].replace('Orissa', 'Odisha')
state_fid['State'] = state_fid['State'].replace('Andaman and Nicobar', 'Andaman and Nicobar Islands')

# Combining map info and numbers

indian_map_data = pd.merge(state_fid, current_stats_statewise, how = 'left', left_on = 'State', right_on = 'State/UnionTerritory')
indian_map_data.fillna(0, inplace = True)
indian_map_data.fid = indian_map_data.fid.astype(str, copy = True)

indian_map_data['Confirmed_Rank'] = indian_map_data.Confirmed.rank(method='max',ascending=False)
indian_map_data['Active_Rank'] = indian_map_data.Active.rank(method='max',ascending=False)
indian_map_data['Recovered_Rank'] = indian_map_data.Recovered.rank(method='max',ascending=False)
indian_map_data['Deceased_Rank'] = indian_map_data.Deceased.rank(method='max',ascending=False)

indian_map_data['hovertext'] = indian_map_data['State']

# Data Table 
data_table = indian_map_data[['State', 'Confirmed', 'Active', 'Recovered', 'Deceased','Confirmed_Rank', 'Active_Rank', 'Recovered_Rank', 'Deceased_Rank']].reset_index(drop=True)

#------- Daily Metrics -------#

all_date_labels = [i.strftime('%b %d') for i in sorted(set(covid_19_india.Date))]
stride = float(str(18/len(all_date_labels))[:4])
date_vals = [round(i*stride,2) for i in range(1,len(all_date_labels)+1)]
date_encoder = dict(zip(date_vals, all_date_labels))

# Cumulative Numbers
metrics_cumulative = daywise.sum().reset_index()
metrics_cumulative.sort_values(by = 'Date', inplace = True)

# Daily Numbers
metrics_cumulative['daily_confirmed'] = metrics_cumulative.Confirmed.diff().fillna(metrics_cumulative.Confirmed)
metrics_cumulative['daily_active'] = metrics_cumulative.Active.diff().fillna(metrics_cumulative.Active)
metrics_cumulative['daily_recovered'] = metrics_cumulative.Recovered.diff().fillna(metrics_cumulative.Recovered)
metrics_cumulative['daily_deceased'] = metrics_cumulative.Deceased.diff().fillna(metrics_cumulative.Deceased)

prev_mode = 'Cumulative'

#--------------------------------- APP START ---------------------------------#
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

server = app.server

buttons = html.Div(
	[
		dbc.Button("Cumulative", id="cum_id", outline=True, color="warning", active=True, className="mr-1"),
		dbc.Button("Daily", id='daily_id', outline=True, color="info", active=True, className="mr-1"),
    ],
)

app.title = "Covid-19 Tracker"

app.layout = html.Div([

	html.H1(children="COVID-19 TRACKER", style={"text-align":"center", "color":" #b48608", "font-family":" Droid serif, serif",
			"font-size":" 50px", "font-weight":" 700", "font-style":" italic", "line-height":" 44px", "margin":" 0 0 12px"}),

	html.Div([
		html.H4("Summary Statistics",style={ "color" : "#c9d0d4", "font-family" : "Merriweather, serif", "font-size":" 30px", 
				"text-align":"center", 'margin-top' : '40px', 'margin-bottom' : '40px'})
	]),

	html.Ul(
		children = [html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Confirmed"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_confirmed
														)
													]
										)
						)
		),
		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Active"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_active
														)
													]
										)
						)
		),
		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Recovered"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_recovered
														)
													]
										)
						)
		),
		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Deceased"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_deceased
														)
													]
										)
						),
			style={'margin-left' : '130px'}
		),

		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Total"), html.Br(), 'Tested']),
													html.P(
														className = "card__num",
														children = latest_tested
														)
													]
										)
						)
		)
		]
	),

    html.Div([
        dcc.Dropdown(
            id='metric',
            options=[{'label': i, 'value': i} for i in current_stats_statewise.columns[1:]],
            placeholder="Select Field",
            value='Confirmed'
        ),

    ],
    	id='india_map_dropdown',
    	style={"font-style":" italic", 'color' : '#f7f3f3', "font-size":" 20px", 
    	'margin-left' : '10px', 'margin-top' : '35px', 'width': '20%'}
    ),

    html.Div([
    	dcc.Graph(id='India_Heat_Map')
	], 
		style={'display':'inline-block', 'width': '50%', 'margin-left' : '10px', 'margin-top' : '15px'}
    ),

    html.Div([
    	html.Pre(
    		id='indian_map_hover_data', 
    	)
	],
		style={ 'display':'inline-block', 'width' : '30%', 'float' : 'right', 
				'padding-top' : '25px', 'margin-top' : '50px', 'margin-right' : '60px'
				}
	),
#------- Data Table -------#

	html.Div([
		dash_table.DataTable(
		    data=data_table.to_dict('records'),
		    columns=[{'id': c, 'name': c} for c in data_table.columns],
		    sort_action="native",
		    sort_mode="multi",
		    style_header={'backgroundColor': 'rgb(30, 30, 30)','fontWeight': 'bold', 'fontSize' : 18, 'border' : '2px solid yellow'},
		    style_cell={'backgroundColor': 'rgb(30, 30, 30)', 'textAlign': 'center', 'fontSize' : 16,
		        		'maWidth' : 0, 'textOverflow': 'ellipsis', 'border' : '1px solid blue'
		        		}
		)
	],
		style={'width' : '97%', 'margin-left' : '21px', 'margin-top' : '40px'}
	),
#------- Daily Metrics -------#

	html.Div([
		dbc.Button("Cumulative", id="cum_id", outline=True, color="warning", active=True, className="mr-1"),
		dbc.Button("Daily", id='daily_id', outline=True, color="info", active=True, className="mr-1")
	],
		style = {'width' : '20%', 'margin-left' : '10px', 'margin-top' : '50px', 'display' : 'inline-block'}
	),


	# html.Div([
	# 	dbc.Button("Daily", id='daily_id', outline=True, color="info", active=True, className="mr-1")
	# ],
	# 	style = {'width' : '7%', 'margin-left' : '15px', 'margin-top' : '50px', 'display' : 'inline-block'}
	# ),

	# html.Div(id='butt_out'),
	

	html.Div([
		html.H5('Choose a date range:')
	],
		style={'color' : 'lime', 'margin-top' : '25px', 'margin-left' : '10px'}
	),

	html.Div([
		dcc.RangeSlider(
			id="metric_date_picker",
			min = min(date_vals),
			max = max(date_vals),
			step = round(date_vals[1]-date_vals[0], 2),
			marks= dict(zip(date_vals[::10], all_date_labels[::10])),
			value = [min(date_vals), max(date_vals)]
		)
	],
		style = {'margin-top' : '20px'}
	),

	html.Div(												# Output of Range Slider
		id='out',
		style={'color' : 'lime', 'margin-left' : '10px'}
	),

	html.Div([
		dcc.Graph(id="confirmed_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid red'
		}
	),

	html.Div([
		dcc.Graph(id="active_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid blue'
		}
	),

	html.Div([
		dcc.Graph(id="recovered_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid green'
		}
	),

	html.Div([
		dcc.Graph(id="deceased_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid grey'
		}
	)
])

#--------------------------------- CALLBACKS ---------------------------------#

				# Update India Map
@app.callback(
	Output('India_Heat_Map', 'figure'),
	[Input('metric', 'value')]
)
def update_graph(metric):

	if metric == 'Confirmed':
		#cs = "reds"
		cs = "inferno"
		rs = True
	elif metric == 'Active':
		#cs = "blues"
		cs = "plotly3"
		rs = False
	elif metric == 'Recovered':
		#cs = "greens"
		cs = "viridis"
		rs = False
	else:
		#cs = "oranges"
		cs = "RdYlGn"
		rs = False

	fig = go.Figure(go.Choroplethmapbox(
						geojson=indian_states, locations=indian_map_data.fid.tolist(), z=indian_map_data[metric].tolist(),
						featureidkey = "id", hovertext=indian_map_data['hovertext'], colorscale=cs, reversescale=rs,
						zmin=indian_map_data[metric].min(), zmax=indian_map_data[metric].max(),marker_opacity=0.8, marker_line_width=1)
					)
	fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=4.22, mapbox_center={"lat": 22.5458, "lon": 82.0882},
						xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True), dragmode=False)
	fig.update_layout(height=800, width=800, margin={"r":0,"t":0,"l":0,"b":0})

	return fig

				# Update India Map Hovertable
@app.callback(
    Output('indian_map_hover_data', 'children'),
    [Input('India_Heat_Map', 'hoverData')])
def update_hover_table(hoverData):
	if hoverData is None:
		return 'Hover over a state for more details'
	hover_state = str(hoverData['points'][0]['hovertext'])
	pd.options.display.width = 50
	pd.options.display.max_colwidth = 50
	hover_df = pd.DataFrame(
		{
			"Field" : ['State', 'Confirmed', 'Confirmed_Rank', 'Active', 'Active_Rank', 'Recovered', 'Recovered_Rank', 'Deceased', 'Deceased_Rank'],
            "Value" : indian_map_data[['State', 'Confirmed', 'Confirmed_Rank', 'Active', 'Active_Rank', 'Recovered', 'Recovered_Rank', 'Deceased', 'Deceased_Rank']][indian_map_data.State == hover_state].values[0].tolist()
        })
	
	hover_table = dbc.Table.from_dataframe(hover_df, striped=True, bordered=True, hover=True, dark=True, responsive=True)
	return hover_table

				# Date Range Output
@app.callback(
	Output('out', 'children'),
	[Input('metric_date_picker','value')]
)
def display(value):
	return 'Chosen date Range: {} to {}'.format(date_encoder[value[0]], date_encoder[value[1]])


@app.callback(
	[Output('confirmed_graph_id', 'figure'),
	 Output('active_graph_id', 'figure'),
	 Output('recovered_graph_id', 'figure'),
	 Output('deceased_graph_id', 'figure')],
	[Input('metric_date_picker', 'value'),
	 Input('cum_id', 'n_clicks'),
	 Input('daily_id', 'n_clicks')
	]
)
def update_daily_metrics(date, cum_time, daily_time):
	start = dt.datetime.strptime(date_encoder[date[0]]+' 2020', '%b %d %Y').strftime("%d-%m-%Y")
	end = dt.datetime.strptime(date_encoder[date[1]]+' 2020', '%b %d %Y').strftime("%d-%m-%Y")

	confirmed_layout = go.Layout(
	    title = dict(text = 'Confirmed Cases', x = 0.5, y = 0.95, font = dict(color = "Red")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='Red', zeroline = False),
	    yaxis = dict(title = '# ' + 'Confirmed', showgrid = False, showline=True, linewidth=2, linecolor='Red', zeroline = False),
	    paper_bgcolor = "#ffcccb",plot_bgcolor = "#ffcccb",legend = {'title' : 'Confirmed'}
	)

	active_layout = layout = go.Layout(
	    title = dict(text = 'Active Cases', x = 0.5, y = 0.95, font = dict(color = "Blue")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='Blue', zeroline = False),
	    yaxis = dict(title = '# ' + 'Active', showgrid = False, showline=True, linewidth=2, linecolor='Blue', zeroline = False),
	    paper_bgcolor = "#87CEFA",plot_bgcolor = "#87CEFA",legend = {'title' : 'Active'}
	)

	recovered_layout = go.Layout(
	    title = dict(text = 'Recovered Cases', x = 0.5, y = 0.95, font = dict(color = "green")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='green', zeroline = False),
	    yaxis = dict(title = '# ' + 'Recovered', showgrid = False, showline=True, linewidth=2, linecolor='green', zeroline = False),
	    paper_bgcolor = "#90EE90",plot_bgcolor = "#90EE90",legend = {'title' : 'Recovered'}
	)

	deceased_layout =go.Layout(
	    title = dict(text = 'Deceased Cases', x = 0.5, y = 0.95, font = dict(color = "grey")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='grey', zeroline = False),
	    yaxis = dict(title = '# ' + 'Deceased', showgrid = False, showline=True, linewidth=2, linecolor='grey', zeroline = False),
	    paper_bgcolor = "#D3D3D3",plot_bgcolor = "#D3D3D3",legend = {'title' : 'Deceased'}
	)

	ctx = dash.callback_context
	
	ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs
    }, indent=2)

	global prev_mode
	mode = prev_mode

	if not ctx.triggered:
		mode = 'Cumulative'
	else:
		val = ctx.triggered[0]['prop_id'].split('.')[0]
		if val == 'cum_id':
			mode = 'Cumulative'
		elif val == 'daily_id':
			mode = 'Daily'
		else:
			mode = prev_mode

	display_type = []
	if mode == 'Cumulative':
		display_type = ['Confirmed', 'Active', 'Recovered', 'Deceased']
	else:
		display_type = ['daily_confirmed', 'daily_active', 'daily_recovered', 'daily_deceased']

	prev_mode = mode

	confirmed_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
            	y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[0]],
            	mode = 'lines+markers', marker = dict(size = 8, color = "red", symbol = "circle",line = dict(width = 2))
			)]

	active_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
                y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[1]],
                mode = 'lines+markers', marker = dict(size = 8, color = "blue", symbol = "star",line = dict(width = 2))
			)]

	recovered_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
                y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[2]],
                mode = 'lines+markers', marker = dict(size = 8, color = "green", symbol = "pentagon",line = dict(width = 2))
			)]

	deceased_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
                y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[3]],
                mode = 'lines+markers', marker = dict(size = 8, color = "gray", symbol = "asterisk",line = dict(width = 2))
			)]

	confirmed_fig = go.Figure(data = confirmed_data, layout = confirmed_layout)
	active_fig = go.Figure(data = active_data, layout = active_layout)
	recovered_fig = go.Figure(data = recovered_data, layout = recovered_layout)
	deceased_fig = go.Figure(data = deceased_data, layout = deceased_layout)

	return confirmed_fig, active_fig, recovered_fig, deceased_fig



if __name__ == '__main__':
=======
import kaggle

import pandas as pd
import numpy as np

import os
import datetime as dt

from urllib.request import urlopen
import json

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import plotly.offline as pyo
import plotly.graph_objs as go

#kaggle.api.authenticate()
#kaggle.api.dataset_download_files('sudalairajkumar/covid19-in-india', path='./covid', unzip=True)

for i in list(map(lambda x: x.split('.csv')[0],os.listdir('./covid'))):
    exec(i + ' = pd.read_csv("./covid/' + i +'.csv")')

#------- Standardising Datasets -------#

def get_ICMRTestingDetails_date(param):
    val = param.split()
    return pd.Series([val[0], val[1]])

ICMRTestingDetails[['Date', 'Time']] = ICMRTestingDetails.DateTime.apply(get_ICMRTestingDetails_date)

covid_19_india = covid_19_india.rename(columns={'Cured' : 'Recovered', 'Deaths' : 'Deceased'})

covid_19_india['Active'] = covid_19_india.Confirmed - (covid_19_india.Deceased + covid_19_india.Recovered)

# Standardising Dates

if str(covid_19_india.Date.iloc[0]).find('/') != -1:
    try:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d/%m/%y')
    except:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d/%m/%Y')
else:
    try:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d-%m-%y')
    except:
        covid_19_india.Date = pd.to_datetime(covid_19_india.Date, format = '%d-%m-%Y')

if str(ICMRTestingDetails.Date.iloc[0]).find('/') != -1:
    try:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d/%m/%y')
    except:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d/%m/%Y')
else:
    try:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d-%m-%y')
    except:
        ICMRTestingDetails.Date = pd.to_datetime(ICMRTestingDetails.Date, format = '%d-%m-%Y')

# Getting Specific Dates

covid_19_recent_date = sorted(set(covid_19_india.Date))[-1]
icmr_recent_date = sorted(set(ICMRTestingDetails.Date))[-1]

today = dt.date.today().strftime('%d-%m-%y')
yesterday = (dt.date.today() - dt.timedelta(1)).strftime('%d-%m-%y')
day_bef_yest = (dt.date.today() - dt.timedelta(2)).strftime('%d-%m-%y')

daywise = covid_19_india.groupby('Date')

def get_current_stats():
    return [daywise.sum().loc[covid_19_recent_date], int(ICMRTestingDetails[ICMRTestingDetails.Date == icmr_recent_date]['TotalSamplesTested'])]

current_stats = get_current_stats()

latest_recovered, latest_deceased, latest_confirmed, latest_active = current_stats[0][['Recovered', 'Deceased', 'Confirmed', 'Active']].values
latest_tested = current_stats[1]

#------- Map Data -------#

current_stats_statewise = covid_19_india.groupby(['Date', 'State/UnionTerritory']).sum().loc[covid_19_recent_date]

state_json_file = open('indian_states_geojson.json')
indian_states = json.load(state_json_file)

for ele in indian_states['features']:
    ele.update({'id' : str(ele['properties']['ID_1'])})

# Extracting the state and feature id's required to plot the map

state_names_ids = []
for ele in indian_states['features']:
    state_names_ids.append([ele['properties']['NAME_1'], ele['id']])

state_fid = pd.DataFrame(state_names_ids, columns = ['State', 'fid'])
state_fid['State'] = state_fid['State'].replace('Uttaranchal', 'Uttarakhand')
state_fid['State'] = state_fid['State'].replace('Telangana', 'Telengana')
state_fid['State'] = state_fid['State'].replace('Orissa', 'Odisha')
state_fid['State'] = state_fid['State'].replace('Andaman and Nicobar', 'Andaman and Nicobar Islands')

# Combining map info and numbers

indian_map_data = pd.merge(state_fid, current_stats_statewise, how = 'left', left_on = 'State', right_on = 'State/UnionTerritory')
indian_map_data.fillna(0, inplace = True)
indian_map_data.fid = indian_map_data.fid.astype(str, copy = True)

indian_map_data['Confirmed_Rank'] = indian_map_data.Confirmed.rank(method='max',ascending=False)
indian_map_data['Active_Rank'] = indian_map_data.Active.rank(method='max',ascending=False)
indian_map_data['Recovered_Rank'] = indian_map_data.Recovered.rank(method='max',ascending=False)
indian_map_data['Deceased_Rank'] = indian_map_data.Deceased.rank(method='max',ascending=False)

indian_map_data['hovertext'] = indian_map_data['State']

# Data Table 
data_table = indian_map_data[['State', 'Confirmed', 'Active', 'Recovered', 'Deceased','Confirmed_Rank', 'Active_Rank', 'Recovered_Rank', 'Deceased_Rank']].reset_index(drop=True)

#------- Daily Metrics -------#

all_date_labels = [i.strftime('%b %d') for i in sorted(set(covid_19_india.Date))]
stride = float(str(18/len(all_date_labels))[:4])
date_vals = [round(i*stride,2) for i in range(1,len(all_date_labels)+1)]
date_encoder = dict(zip(date_vals, all_date_labels))

# Cumulative Numbers
metrics_cumulative = daywise.sum().reset_index()
metrics_cumulative.sort_values(by = 'Date', inplace = True)

# Daily Numbers
metrics_cumulative['daily_confirmed'] = metrics_cumulative.Confirmed.diff().fillna(metrics_cumulative.Confirmed)
metrics_cumulative['daily_active'] = metrics_cumulative.Active.diff().fillna(metrics_cumulative.Active)
metrics_cumulative['daily_recovered'] = metrics_cumulative.Recovered.diff().fillna(metrics_cumulative.Recovered)
metrics_cumulative['daily_deceased'] = metrics_cumulative.Deceased.diff().fillna(metrics_cumulative.Deceased)

prev_mode = 'Cumulative'

#--------------------------------- APP START ---------------------------------#
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

server = app.server

buttons = html.Div(
	[
		dbc.Button("Cumulative", id="cum_id", outline=True, color="warning", active=True, className="mr-1"),
		dbc.Button("Daily", id='daily_id', outline=True, color="info", active=True, className="mr-1"),
    ],
)

app.title = "Covid-19 Tracker"

app.layout = html.Div([

	html.H1(children="COVID-19 TRACKER", style={"text-align":"center", "color":" #b48608", "font-family":" Droid serif, serif",
			"font-size":" 50px", "font-weight":" 700", "font-style":" italic", "line-height":" 44px", "margin":" 0 0 12px"}),

	html.Div([
		html.H4("Summary Statistics",style={ "color" : "#c9d0d4", "font-family" : "Merriweather, serif", "font-size":" 30px", 
				"text-align":"center", 'margin-top' : '40px', 'margin-bottom' : '40px'})
	]),

	html.Ul(
		children = [html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Confirmed"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_confirmed
														)
													]
										)
						)
		),
		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Active"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_active
														)
													]
										)
						)
		),
		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Recovered"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_recovered
														)
													]
										)
						)
		),
		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Deceased"), html.Br(), 'Cases']),
													html.P(
														className = "card__num",
														children = latest_deceased
														)
													]
										)
						),
			style={'margin-left' : '130px'}
		),

		html.Li(
			className = "card", 
			children = html.Div(
							className = "card__flipper",
							children = html.Div(
											className = "card__front",
											children = [
													html.P(
														className = "card__name", 
														children = [html.Span("Total"), html.Br(), 'Tested']),
													html.P(
														className = "card__num",
														children = latest_tested
														)
													]
										)
						)
		)
		]
	),

    html.Div([
        dcc.Dropdown(
            id='metric',
            options=[{'label': i, 'value': i} for i in current_stats_statewise.columns[1:]],
            placeholder="Select Field",
            value='Confirmed'
        ),

    ],
    	id='india_map_dropdown',
    	style={"font-style":" italic", 'color' : '#f7f3f3', "font-size":" 20px", 
    	'margin-left' : '10px', 'margin-top' : '35px', 'width': '20%'}
    ),

    html.Div([
    	dcc.Graph(id='India_Heat_Map')
	], 
		style={'display':'inline-block', 'width': '50%', 'margin-left' : '10px', 'margin-top' : '15px'}
    ),

    html.Div([
    	html.Pre(
    		id='indian_map_hover_data', 
    	)
	],
		style={ 'display':'inline-block', 'width' : '30%', 'float' : 'right', 
				'padding-top' : '25px', 'margin-top' : '50px', 'margin-right' : '60px'
				}
	),
#------- Data Table -------#

	html.Div([
		dash_table.DataTable(
		    data=data_table.to_dict('records'),
		    columns=[{'id': c, 'name': c} for c in data_table.columns],
		    sort_action="native",
		    sort_mode="multi",
		    style_header={'backgroundColor': 'rgb(30, 30, 30)','fontWeight': 'bold', 'fontSize' : 18, 'border' : '2px solid yellow'},
		    style_cell={'backgroundColor': 'rgb(30, 30, 30)', 'textAlign': 'center', 'fontSize' : 16,
		        		'maWidth' : 0, 'textOverflow': 'ellipsis', 'border' : '1px solid blue'
		        		}
		)
	],
		style={'width' : '97%', 'margin-left' : '21px', 'margin-top' : '40px'}
	),
#------- Daily Metrics -------#

	html.Div([
		dbc.Button("Cumulative", id="cum_id", outline=True, color="warning", active=True, className="mr-1"),
		dbc.Button("Daily", id='daily_id', outline=True, color="info", active=True, className="mr-1")
	],
		style = {'width' : '20%', 'margin-left' : '10px', 'margin-top' : '50px', 'display' : 'inline-block'}
	),


	# html.Div([
	# 	dbc.Button("Daily", id='daily_id', outline=True, color="info", active=True, className="mr-1")
	# ],
	# 	style = {'width' : '7%', 'margin-left' : '15px', 'margin-top' : '50px', 'display' : 'inline-block'}
	# ),

	# html.Div(id='butt_out'),
	

	html.Div([
		html.H5('Choose a date range:')
	],
		style={'color' : 'lime', 'margin-top' : '25px', 'margin-left' : '10px'}
	),

	html.Div([
		dcc.RangeSlider(
			id="metric_date_picker",
			min = min(date_vals),
			max = max(date_vals),
			step = round(date_vals[1]-date_vals[0], 2),
			marks= dict(zip(date_vals[::10], all_date_labels[::10])),
			value = [min(date_vals), max(date_vals)]
		)
	],
		style = {'margin-top' : '20px'}
	),

	html.Div(												# Output of Range Slider
		id='out',
		style={'color' : 'lime', 'margin-left' : '10px'}
	),

	html.Div([
		dcc.Graph(id="confirmed_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid red'
		}
	),

	html.Div([
		dcc.Graph(id="active_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid blue'
		}
	),

	html.Div([
		dcc.Graph(id="recovered_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid green'
		}
	),

	html.Div([
		dcc.Graph(id="deceased_graph_id")
	],
		style={
			'margin-top' : '50px',
			'border':'3px solid grey'
		}
	)
])

#--------------------------------- CALLBACKS ---------------------------------#

				# Update India Map
@app.callback(
	Output('India_Heat_Map', 'figure'),
	[Input('metric', 'value')]
)
def update_graph(metric):

	if metric == 'Confirmed':
		#cs = "reds"
		cs = "inferno"
		rs = True
	elif metric == 'Active':
		#cs = "blues"
		cs = "plotly3"
		rs = False
	elif metric == 'Recovered':
		#cs = "greens"
		cs = "viridis"
		rs = False
	else:
		#cs = "oranges"
		cs = "RdYlGn"
		rs = False

	fig = go.Figure(go.Choroplethmapbox(
						geojson=indian_states, locations=indian_map_data.fid.tolist(), z=indian_map_data[metric].tolist(),
						featureidkey = "id", hovertext=indian_map_data['hovertext'], colorscale=cs, reversescale=rs,
						zmin=indian_map_data[metric].min(), zmax=indian_map_data[metric].max(),marker_opacity=0.8, marker_line_width=1)
					)
	fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=4.22, mapbox_center={"lat": 22.5458, "lon": 82.0882},
						xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True), dragmode=False)
	fig.update_layout(height=800, width=800, margin={"r":0,"t":0,"l":0,"b":0})

	return fig

				# Update India Map Hovertable
@app.callback(
    Output('indian_map_hover_data', 'children'),
    [Input('India_Heat_Map', 'hoverData')])
def update_hover_table(hoverData):
	if hoverData is None:
		return 'Hover over a state for more details'
	hover_state = str(hoverData['points'][0]['hovertext'])
	pd.options.display.width = 50
	pd.options.display.max_colwidth = 50
	hover_df = pd.DataFrame(
		{
			"Field" : ['State', 'Confirmed', 'Confirmed_Rank', 'Active', 'Active_Rank', 'Recovered', 'Recovered_Rank', 'Deceased', 'Deceased_Rank'],
            "Value" : indian_map_data[['State', 'Confirmed', 'Confirmed_Rank', 'Active', 'Active_Rank', 'Recovered', 'Recovered_Rank', 'Deceased', 'Deceased_Rank']][indian_map_data.State == hover_state].values[0].tolist()
        })
	
	hover_table = dbc.Table.from_dataframe(hover_df, striped=True, bordered=True, hover=True, dark=True, responsive=True)
	return hover_table

				# Date Range Output
@app.callback(
	Output('out', 'children'),
	[Input('metric_date_picker','value')]
)
def display(value):
	return 'Chosen date Range: {} to {}'.format(date_encoder[value[0]], date_encoder[value[1]])


@app.callback(
	[Output('confirmed_graph_id', 'figure'),
	 Output('active_graph_id', 'figure'),
	 Output('recovered_graph_id', 'figure'),
	 Output('deceased_graph_id', 'figure')],
	[Input('metric_date_picker', 'value'),
	 Input('cum_id', 'n_clicks'),
	 Input('daily_id', 'n_clicks')
	]
)
def update_daily_metrics(date, cum_time, daily_time):
	start = dt.datetime.strptime(date_encoder[date[0]]+' 2020', '%b %d %Y').strftime("%d-%m-%Y")
	end = dt.datetime.strptime(date_encoder[date[1]]+' 2020', '%b %d %Y').strftime("%d-%m-%Y")

	confirmed_layout = go.Layout(
	    title = dict(text = 'Confirmed Cases', x = 0.5, y = 0.95, font = dict(color = "Red")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='Red', zeroline = False),
	    yaxis = dict(title = '# ' + 'Confirmed', showgrid = False, showline=True, linewidth=2, linecolor='Red', zeroline = False),
	    paper_bgcolor = "#ffcccb",plot_bgcolor = "#ffcccb",legend = {'title' : 'Confirmed'}
	)

	active_layout = layout = go.Layout(
	    title = dict(text = 'Active Cases', x = 0.5, y = 0.95, font = dict(color = "Blue")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='Blue', zeroline = False),
	    yaxis = dict(title = '# ' + 'Active', showgrid = False, showline=True, linewidth=2, linecolor='Blue', zeroline = False),
	    paper_bgcolor = "#87CEFA",plot_bgcolor = "#87CEFA",legend = {'title' : 'Active'}
	)

	recovered_layout = go.Layout(
	    title = dict(text = 'Recovered Cases', x = 0.5, y = 0.95, font = dict(color = "green")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='green', zeroline = False),
	    yaxis = dict(title = '# ' + 'Recovered', showgrid = False, showline=True, linewidth=2, linecolor='green', zeroline = False),
	    paper_bgcolor = "#90EE90",plot_bgcolor = "#90EE90",legend = {'title' : 'Recovered'}
	)

	deceased_layout =go.Layout(
	    title = dict(text = 'Deceased Cases', x = 0.5, y = 0.95, font = dict(color = "grey")),
	    xaxis = dict(title = 'Days', showgrid = False, showline=True, linewidth=2, linecolor='grey', zeroline = False),
	    yaxis = dict(title = '# ' + 'Deceased', showgrid = False, showline=True, linewidth=2, linecolor='grey', zeroline = False),
	    paper_bgcolor = "#D3D3D3",plot_bgcolor = "#D3D3D3",legend = {'title' : 'Deceased'}
	)

	ctx = dash.callback_context
	
	ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs
    }, indent=2)

	global prev_mode
	mode = prev_mode

	if not ctx.triggered:
		mode = 'Cumulative'
	else:
		val = ctx.triggered[0]['prop_id'].split('.')[0]
		if val == 'cum_id':
			mode = 'Cumulative'
		elif val == 'daily_id':
			mode = 'Daily'
		else:
			mode = prev_mode

	display_type = []
	if mode == 'Cumulative':
		display_type = ['Confirmed', 'Active', 'Recovered', 'Deceased']
	else:
		display_type = ['daily_confirmed', 'daily_active', 'daily_recovered', 'daily_deceased']

	prev_mode = mode

	confirmed_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
            	y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[0]],
            	mode = 'lines+markers', marker = dict(size = 8, color = "red", symbol = "circle",line = dict(width = 2))
			)]

	active_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
                y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[1]],
                mode = 'lines+markers', marker = dict(size = 8, color = "blue", symbol = "star",line = dict(width = 2))
			)]

	recovered_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
                y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[2]],
                mode = 'lines+markers', marker = dict(size = 8, color = "green", symbol = "pentagon",line = dict(width = 2))
			)]

	deceased_data = [go.Scatter(
				x = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))]['Date'],
                y = metrics_cumulative[(metrics_cumulative['Date'] >= pd.to_datetime(start, format = "%d-%m-%Y")) & (metrics_cumulative['Date'] <= pd.to_datetime(end, format = "%d-%m-%Y"))][display_type[3]],
                mode = 'lines+markers', marker = dict(size = 8, color = "gray", symbol = "asterisk",line = dict(width = 2))
			)]

	confirmed_fig = go.Figure(data = confirmed_data, layout = confirmed_layout)
	active_fig = go.Figure(data = active_data, layout = active_layout)
	recovered_fig = go.Figure(data = recovered_data, layout = recovered_layout)
	deceased_fig = go.Figure(data = deceased_data, layout = deceased_layout)

	return confirmed_fig, active_fig, recovered_fig, deceased_fig



if __name__ == '__main__':
>>>>>>> d3be27fe83dd3f6ca5312aeaf578e0c62b2c0abb
	app.run_server(debug = True)