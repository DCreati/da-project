import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import os
import plotly.graph_objects as go

# Dati tab: Percentuale
disinfo_map = {0: 'truth', 1: 'disinformation'}

dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'liar_dataset.csv')
liar_dataset = pd.read_csv(dataset_path)
liar_dataset['disinfo_text'] = liar_dataset['disinfo'].map(disinfo_map)

disinfo_counts = liar_dataset[liar_dataset['disinfo'] == 1]['speaker'].value_counts().sort_values(ascending=False)
speaker_order = disinfo_counts.index.tolist()

agg = liar_dataset.groupby('speaker').agg(
    total=('label', 'size'),
    disinfo_count=('disinfo', 'sum')
).query('total >= 50')
agg['disinfo_ratio'] = agg['disinfo_count'] / agg['total']
agg_sorted = agg.sort_values(by='disinfo_ratio', ascending=False)
pivot = agg_sorted[['disinfo_ratio']]

fig2 = px.imshow(
    pivot.T,
    text_auto='.2f',
    labels={'x': 'Speaker', 'y': ''},
    color_continuous_scale='Reds',
    aspect='auto',
    title='Proporzione di disinformazione per speaker'
)
fig2.update_yaxes(showticklabels=False)

partial_df = liar_dataset[['label', 'speaker', 'date', 'credibility_score']].copy()
partial_df['date'] = pd.to_datetime(partial_df['date'])
partial_df['year'] = partial_df['date'].dt.year
partial_df = partial_df.drop(columns=['date'])
partial_df = partial_df[partial_df['year'] >= 2007]

# Dati tab: Veridicità
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

#date_min = daily_mean['date'].min()
date_min = pd.to_datetime('01-01-2007')
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

#Dati tab: Contesto
disinfo_counts_context = liar_dataset[liar_dataset['disinfo'] == 1]['context'].value_counts().sort_values(ascending=False)
context_order = disinfo_counts_context.index.tolist()

agg_context = liar_dataset.groupby('context').agg(
    total=('label', 'size'),
    disinfo_count=('disinfo', 'sum')
).query('total >= 50')
agg_context['disinfo_ratio'] = agg_context['disinfo_count'] / agg_context['total']
agg_context_sorted = agg_context.sort_values(by='disinfo_ratio', ascending=False)
pivot_context = agg_context_sorted[['disinfo_ratio']]

fig_context_heatmap = px.imshow(
    pivot_context.T,
    text_auto='.2f',
    labels={'x': 'Contesto', 'y': ''},
    color_continuous_scale='Reds',
    aspect='auto',
    title='Proporzione di disinformazione per contesto'
)
fig_context_heatmap.update_yaxes(showticklabels=False)

# Dati tab: Subject
disinfo_counts_subject = liar_dataset[liar_dataset['disinfo'] == 1]['subject'].value_counts().sort_values(ascending=False)
subject_order = disinfo_counts_subject.index.tolist()

agg_subject = liar_dataset.groupby("subject").agg(
    total=("label", "size"),
    disinfo_count=("disinfo", "sum")
).query("total >= 50")
agg_subject["disinfo_ratio"] = agg_subject["disinfo_count"] / agg_subject["total"]
agg_subject_sorted = agg_subject.sort_values(by="disinfo_ratio", ascending=False)

pivot_subject = agg_subject_sorted[["disinfo_ratio"]]

fig_subject_heatmap = px.imshow(
    pivot_subject.T,
    text_auto=".2f",
    labels={"x": "Tema", "y": ""},
    color_continuous_scale="Reds",
    aspect="auto",
    title="Proporzione di disinformazione per tema"
)
fig_subject_heatmap.update_yaxes(showticklabels=False)

# Gestione app per dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = dbc.Container(
    [
        html.H1('Evoluzione nel tempo delle notizie', className='my-4'),
        dbc.Tabs(
            [
                dbc.Tab(label='Frequenza', tab_id='scatter'),
                dbc.Tab(label='Percentuale', tab_id='line'),
                dbc.Tab(label='Veridicità', tab_id='veridicita'),
                dbc.Tab(label='Contesto', tab_id='context'),
                dbc.Tab(label='Tema', tab_id='subject'),
            ],
            id='tabs',
            active_tab='scatter',
        ),
        html.Div(id='tab-content', className='p-4'),
    ],
    fluid=True
)

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_tab_content(active_tab):
    if active_tab == 'scatter':
        return html.Div([
            dcc.Graph(id='graph-with-slider'),
            dcc.Slider(
                id='year-slider',
                min=partial_df['year'].min(),
                max=partial_df['year'].max(),
                value=partial_df['year'].min(),
                marks={str(year): str(year) for year in sorted(partial_df['year'].unique())},
                step=None
            )
        ])
    elif active_tab == 'line':
        return html.Div([
            html.Label('Seleziona uno speaker:'),
            dcc.Dropdown(
                id='speaker-dropdown',
                options=[
                    {'label': speaker, 'value': speaker}
                    for speaker in sorted(partial_df['speaker'].unique())
                ],
                value=partial_df['speaker'].value_counts().idxmax(),  # default: speaker più frequente
                clearable=False,
                style={'width': '50%'}
            ),
            dcc.Graph(id='speaker-fake-news-graph'),
            html.Hr(),
            html.Label('Numero di speaker da mostrare:'),
            dcc.Input(
                id='num-speakers',
                type='number',
                min=1,
                max=len(speaker_order),
                value=10,
                step=1,
                style={'marginBottom': '20px'}
            ),
            dcc.Graph(id='speaker-histogram'),
            dcc.Graph(figure=fig2)
        ])
    elif active_tab == 'veridicita':
        return html.Div([
            html.H4('Analisi della Veridicità', style={'textAlign': 'center'}),
            dcc.Graph(id='veridicita-graph'),
            html.Br(),
            dcc.RangeSlider(
                id='date-slider',
                min=0,
                max=date_range,
                value=[0, date_range],
                marks=marks,
                allowCross=False,
                tooltip={'placement': 'bottom', 'always_visible': False},
                step=1
            ),
            html.Div(id='selected-date-range', style={'textAlign': 'center', 'marginTop': 10})
        ])
    elif active_tab == 'context':
        return html.Div([
            html.H4('Analisi della Disinformazione per Contesto', style={'textAlign': 'center'}),
            html.Label('Numero di contesti da mostrare:'),
            dcc.Input(
                id='num-contexts',
                type='number',
                min=1,
                max=len(context_order),
                value=10,
                step=1,
                style={'marginBottom': '20px'}
            ),
            dcc.Graph(id='context-histogram'),
            dcc.Graph(figure=fig_context_heatmap)
        ])
    elif active_tab == 'subject':
        return html.Div([
            html.H4('Analisi della Disinformazione per Tema', style={'textAlign': 'center'}),
            html.Label('Numero di temi da mostrare:'),
            dcc.Input(
                id='num-subjects',
                type='number',
                min=1,
                max=len(subject_order),
                value=10,
                step=1,
                style={'marginBottom': '20px'}
            ),
            dcc.Graph(id='subject-histogram'),
            dcc.Graph(figure=fig_subject_heatmap)
        ])
    return html.Div('Tab non trovata.')

@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value')
)
def update_figure(selected_year):
    filtered_df = partial_df[partial_df['year'] == selected_year]

    speaker_counts = (
        filtered_df['speaker']
        .value_counts()
        .head(30)
        .reset_index()
    )
    speaker_counts.columns = ['speaker', 'frequenza']

    speaker_counts = speaker_counts.merge(
        filtered_df[['speaker', 'credibility_score']],
        on='speaker',
        how='left'
    ).drop_duplicates()

    speaker_counts = speaker_counts.sort_values(by='frequenza', ascending=True)

    fig = px.scatter(
        speaker_counts,
        x='frequenza',
        y='credibility_score',
        title=f'Frequenza delle dichiarazioni per i primi 30 speaker ({selected_year})',
        labels={'frequenza': 'Frequenza', 'speaker': 'Speaker', 'credibility_score': 'Credibility Score'},
        height=700,
        size='frequenza',
        size_max=50,
        hover_name='speaker',
    )

    fig.update_layout(
        yaxis=dict(tickfont=dict(size=10)),
        coloraxis_showscale=False
    )

    return fig

@app.callback(
    Output('speaker-fake-news-graph', 'figure'),
    Input('speaker-dropdown', 'value')
)
def update_speaker_graph(selected_speaker):
    df_speaker = partial_df[partial_df['speaker'] == selected_speaker]

    total_per_year = df_speaker.groupby('year').size().rename('total_statements')
    fake_per_year = df_speaker[df_speaker['label'].isin([0, 1, 2])].groupby('year').size().rename('fake_statements')

    df_plot = pd.concat([total_per_year, fake_per_year], axis=1).fillna(0)
    df_plot['percent_fake'] = (df_plot['fake_statements'] / df_plot['total_statements']) * 100
    df_plot = df_plot.reset_index()

    fig = px.line(
        df_plot,
        x='year',
        y='percent_fake',
        title=f'Percentuale di fake news per anno - {selected_speaker}',
        labels={
            'year': 'Anno',
            'percent_fake': '% Notizie False',
            'total_statements': 'Numero Statement'
        },
        markers=True,
        hover_data={
            'year': True,
            'percent_fake': ':.2f',
            'total_statements': True
        }
    )
    fig.update_layout(yaxis_range=[-10, 110])

    return fig

@app.callback(
    Output('speaker-histogram', 'figure'),
    Input('num-speakers', 'value')
)
def update_histogram(num_speakers):
    limited_speakers = speaker_order[:num_speakers]
    filtered = liar_dataset[liar_dataset['speaker'].isin(limited_speakers)]
    fig = px.histogram(
        filtered,
        x='speaker',
        color='disinfo_text',
        barmode='group',
        category_orders={'disinfo_text': ['disinformation', 'truth'], 'speaker': limited_speakers},
        labels={'disinfo_text': 'Disinformazione', 'speaker': 'Speaker', 'count': 'Numero di statement'},
        title=f'Conteggio per speaker: disinformazione vs verità/parzialità (top {num_speakers})'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        yaxis_title='Numero di statement'
    )
    return fig

@app.callback(
    [Output('veridicita-graph', 'figure'),
     Output('selected-date-range', 'children')],
    [Input('date-slider', 'value')]
)
def update_veridicita_graph(slider_range):
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
        title='Veridicità media e frequenza affermazioni nel tempo (media mobile 30 giorni)',
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

@app.callback(
    Output('context-histogram', 'figure'),
    Input('num-contexts', 'value')
)
def update_context_histogram(num_contexts):
    limited_contexts = context_order[:num_contexts]
    filtered = liar_dataset[liar_dataset['context'].isin(limited_contexts)].copy()

    fig = px.histogram(
        filtered,
        x='context',
        color='disinfo_text',
        barmode='group',
        category_orders={'disinfo_text': ['Disinformazione', 'Verità'], 'context': limited_contexts},
        labels={
            'disinfo_text': 'Tipo',
            'context': 'Contesto',
            'count': 'Numero di statement'
        },
        title=f'Conteggio per contesto: disinformazione vs verità (top {num_contexts})'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        yaxis_title='Numero di statement'
    )

    return fig

@app.callback(
    Output('subject-histogram', 'figure'),
    Input('num-subjects', 'value')
)
def update_subject_histogram(num_subjects):
    limited_subjects = subject_order[:num_subjects]
    filtered = liar_dataset[liar_dataset['subject'].isin(limited_subjects)]
    fig = px.histogram(
        filtered,
        x="subject",
        color="disinfo_text",
        barmode="group",
        category_orders={"disinfo_text": ['disinformation', 'truth'], "subject": limited_subjects},
        labels={"disinfo_text": "Disinformazione", "subject": "Tema"},
        title=f"Conteggio per tema: disinformazione vs verità/parzialità (top {num_subjects})"
    )
    fig.update_layout(xaxis_tickangle=-45, height=600, yaxis_title='Numero di statement')
    return fig


if __name__ == '__main__':
    app.run(debug=True)
