import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import os

dataset_path = os.path.join(os.path.dirname(__file__), '../../data/liar_dataset.csv')
liar_dataset = pd.read_csv(dataset_path)

liar_dataset['date'] = pd.to_datetime(liar_dataset['date'], errors='coerce')
liar_dataset = liar_dataset.dropna(subset=['date'])

daily_mean = liar_dataset.groupby('date').agg(label_mean=('label', 'mean')).reset_index()
daily_mean['rolling_mean_30d'] = daily_mean['label_mean'].rolling(window=30, min_periods=10).mean()

liar_dataset['count'] = 1
daily_counts = liar_dataset.groupby('date').agg(n_statements=('count', 'sum')).reset_index()
daily_counts['rolling_count_30d'] = daily_counts['n_statements'].rolling(window=30, min_periods=10).mean()

merged = pd.merge(
    daily_mean[['date', 'rolling_mean_30d']],
    daily_counts[['date', 'rolling_count_30d']],
    on='date',
    how='inner'
)

import plotly.graph_objects as go

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=merged['date'],
    y=merged['rolling_mean_30d'],
    mode='lines',
    name='Media Veridicità',
    line=dict(color='royalblue')
))
fig1.add_trace(go.Scatter(
    x=merged['date'],
    y=merged['rolling_count_30d'],
    mode='lines',
    name='N. medio affermazioni',
    line=dict(color='orange'),
    yaxis='y2'
))
fig1.update_layout(
    title='📈 Veridicità media e frequenza affermazioni nel tempo (media mobile 30 giorni)',
    xaxis_title='Data',
    yaxis=dict(
        title=dict(text='Media Veridicità', font=dict(color='royalblue')),
        tickfont=dict(color='royalblue')
    ),
    yaxis2=dict(
        title=dict(text='N. medio affermazioni', font=dict(color='orange')),
        tickfont=dict(color='orange'),
        overlaying='y',
        side='right'
    ),
    template='plotly'
)

date_min = daily_mean['date'].min()
date_max = daily_mean['date'].max()
date_range = (date_max - date_min).days

marks = {}
current = date_min
while current <= date_max:
    days = (current - date_min).days
    marks[days] = current.strftime('%Y-%m-%d')
    current += pd.DateOffset(years=1)
marks[0] = date_min.strftime('%Y-%m-%d')
marks[date_range] = date_max.strftime('%Y-%m-%d')

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Analisi della Veridicità", style={'textAlign': 'center'}),
    dcc.Graph(id='veridicita-graph', figure=fig1),
    html.Br(),
    dcc.RangeSlider(
        id='date-slider',
        min=0,
        max=date_range,
        value=[0, date_range],
        marks=marks,
        allowCross=False,
        tooltip={"placement": "bottom", "always_visible": False},  # Nasconde il tooltip nativo
        step=1
    ),
    html.Div(id='selected-date-range', style={'textAlign': 'center', 'marginTop': 10})
], style={'padding': '20px'})

@app.callback(
    [Output('veridicita-graph', 'figure'),
     Output('selected-date-range', 'children')],
    [Input('date-slider', 'value')]
)
def update_graph(slider_range):
    start_date = date_min + pd.Timedelta(days=slider_range[0])
    end_date = date_min + pd.Timedelta(days=slider_range[1])
    filtered = merged[
        (merged['date'] >= start_date) &
        (merged['date'] <= end_date)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered['date'],
        y=filtered['rolling_mean_30d'],
        mode='lines',
        name='Media Veridicità',
        line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        x=filtered['date'],
        y=filtered['rolling_count_30d'],
        mode='lines',
        name='N. medio affermazioni',
        line=dict(color='orange'),
        yaxis='y2'
    ))
    fig.update_layout(
        title='📈 Veridicità media e frequenza affermazioni nel tempo (media mobile 30 giorni)',
        xaxis_title='Data',
        yaxis=dict(
            title=dict(text='Media Veridicità', font=dict(color='royalblue')),
            tickfont=dict(color='royalblue')
        ),
        yaxis2=dict(
            title=dict(text='N. medio affermazioni', font=dict(color='orange')),
            tickfont=dict(color='orange'),
            overlaying='y',
            side='right'
        ),
        template='plotly'
    )
    range_text = f"Intervallo selezionato: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"
    return fig, range_text

if __name__ == '__main__':
    app.run(debug=True)
