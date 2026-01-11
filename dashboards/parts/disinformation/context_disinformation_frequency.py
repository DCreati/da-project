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

disinfo_counts = liar_dataset[liar_dataset['disinfo'] == 1]['context'].value_counts().sort_values(ascending=False)
context_order = disinfo_counts.index.tolist()

agg = liar_dataset.groupby("context").agg(
    total=("label", "size"),
    disinfo_count=("disinfo", "sum")
).query("total >= 50")
agg["disinfo_ratio"] = agg["disinfo_count"] / agg["total"]
agg_sorted = agg.sort_values(by="disinfo_ratio", ascending=False)
pivot = agg_sorted[["disinfo_ratio"]]

fig2 = px.imshow(
    pivot.T,
    text_auto=".2f",
    labels={"x": "Contesto", "y": ""},
    color_continuous_scale="Reds",
    aspect="auto",
    title="🔥 Proporzione di disinformazione per contesto"
)
fig2.update_yaxes(showticklabels=False)

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Dashboard Disinformazione - Contesto (Context)", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Numero di contesti da mostrare:"),
        dcc.Input(
            id='num-contexts',
            type='number',
            min=1,
            max=len(context_order),
            value=10,
            step=1,
            style={'marginBottom': '20px'}
        ),
        dcc.Graph(id='context-histogram')
    ], style={'marginBottom': '50px'}),

    html.Div([
        dcc.Graph(figure=fig2)
    ])
], style={'padding': '30px'})

@app.callback(
    Output('context-histogram', 'figure'),
    Input('num-contexts', 'value')
)
def update_histogram(num_contexts):
    limited_contexts = context_order[:num_contexts]
    filtered = liar_dataset[liar_dataset['context'].isin(limited_contexts)]
    fig = px.histogram(
        filtered,
        x="context",
        color="disinfo_text",
        barmode="group",
        category_orders={"disinfo_text": ['disinfo', 'truth'], "context": limited_contexts},
        labels={"disinfo_text": "Disinformazione", "context": "Contesto"},
        title=f"📊 Conteggio per contesto: disinformazione vs verità/parzialità (top {num_contexts})"
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    return fig

if __name__ == '__main__':
    app.run(debug=True)