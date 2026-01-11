import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import os

disinfo_map = {0: 'truth', 1: 'disinformation'}
disinfo_convert = {'truth': 0, 'disinformation': 1}

dataset_path = os.path.join(os.path.dirname(__file__), '../../../data/liar_dataset.csv')
liar_dataset = pd.read_csv(dataset_path)

liar_dataset['disinfo_text'] = liar_dataset['disinfo'].map(disinfo_map)

disinfo_counts = liar_dataset[liar_dataset['disinfo'] == 1]['subject'].value_counts().sort_values(ascending=False)
subject_order = disinfo_counts.index.tolist()

fig1 = px.histogram(
    liar_dataset,
    x="subject",
    color="disinfo_text",
    barmode="group",
    category_orders={"disinfo_text": ['disinfo', 'truth'], "subject": subject_order},
    labels={"disinfo_text": "Disinformazione", "subject": "Tema"},
    title="📊 Conteggio per tema: disinformazione (1) vs verità/parzialità (0)"
)
fig1.update_layout(xaxis_tickangle=-45, height=600)

agg = liar_dataset.groupby("subject").agg(
    total=("label", "size"),
    disinfo_count=("disinfo", "sum")
).query("total >= 50")
agg["disinfo_ratio"] = agg["disinfo_count"] / agg["total"]
agg_sorted = agg.sort_values(by="disinfo_ratio", ascending=False)

pivot = agg_sorted[["disinfo_ratio"]]

fig2 = px.imshow(
    pivot.T,
    text_auto=".2f",
    labels={"x": "Tema", "y": ""},
    color_continuous_scale="Reds",
    aspect="auto",
    title="🔥 Proporzione di disinformazione per tema"
)
fig2.update_yaxes(showticklabels=False)  # nasconde l'etichetta asse Y

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Dashboard Disinformazione - Tema (Subject)", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Numero di temi da mostrare:"),
        dcc.Input(
            id='num-subjects',
            type='number',
            min=1,
            max=len(subject_order),
            value=10,
            step=1,
            style={'marginBottom': '20px'}
        ),
        dcc.Graph(id='subject-histogram')
    ], style={'marginBottom': '50px'}),

    html.Div([
        html.Label("Numero di temi da mostrare nella heatmap:"),
        dcc.Input(
            id='num-subjects-heatmap',
            type='number',
            min=1,
            max=len(subject_order),
            value=10,
            step=1,
            style={'marginBottom': '20px'}
        ),
        dcc.Graph(id='subject-heatmap')
    ])
], style={'padding': '30px'})

@app.callback(
    Output('subject-histogram', 'figure'),
    Input('num-subjects', 'value')
)
def update_histogram(num_subjects):
    limited_subjects = subject_order[:num_subjects]
    filtered = liar_dataset[liar_dataset['subject'].isin(limited_subjects)]
    fig = px.histogram(
        filtered,
        x="subject",
        color="disinfo_text",
        barmode="group",
        category_orders={"disinfo_text": ['disinfo', 'truth'], "subject": limited_subjects},
        labels={"disinfo_text": "Disinformazione", "subject": "Tema"},
        title=f"📊 Conteggio per tema: disinformazione vs verità/parzialità (top {num_subjects})"
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    return fig

@app.callback(
    Output('subject-heatmap', 'figure'),
    Input('num-subjects-heatmap', 'value')
)
def update_heatmap(num_subjects_heatmap):
    limited_subjects = subject_order[:num_subjects_heatmap]
    agg = liar_dataset[liar_dataset['subject'].isin(limited_subjects)].groupby("subject").agg(
        total=("label", "size"),
        disinfo_count=("disinfo", "sum")
    ).query("total >= 50")
    agg["disinfo_ratio"] = agg["disinfo_count"] / agg["total"]
    agg_sorted = agg.sort_values(by="disinfo_ratio", ascending=False)
    pivot = agg_sorted[["disinfo_ratio"]]
    fig = px.imshow(
        pivot.T,
        text_auto=".2f",
        labels={"x": "Tema", "y": ""},
        color_continuous_scale="Reds",
        aspect="auto",
        title="🔥 Proporzione di disinformazione per tema"
    )
    fig.update_yaxes(showticklabels=False)
    return fig

if __name__ == '__main__':
    app.run(debug=True)