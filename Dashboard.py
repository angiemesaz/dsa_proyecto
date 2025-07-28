
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import requests

# Cargar datos y modelo
df = pd.read_csv("abt.csv")
# Convertir 'ym' (formato YYYYMM) a datetime
df["ym"] = pd.to_datetime(df["ym"].astype(str), format="%Y%m")
df['ym'] = pd.to_datetime(df['ym'])
df['ym_str'] = df['ym'].dt.strftime('%Y-%m')

# App
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard de Ventas"

# Layout
app.layout = html.Div([
    html.H1("Dashboard de Ventas de Marca", style={"textAlign": "center"}),

    html.Div([
        html.Label("Gerencia"),
        dcc.Dropdown(df["Gerencia"].unique(), id="filtro-gerencia", value=df["Gerencia"].unique()[0])
    ], style={"width": "24%", "display": "inline-block"}),

    html.Div([
        html.Label("Segmento"),
        dcc.Dropdown(df["subcanal"].unique(), id="filtro-segmento", value=df["subcanal"].unique()[0])
    ], style={"width": "24%", "display": "inline-block"}),

    html.Div([
        html.Label("Marca"),
        dcc.Dropdown(df["brand"].unique(), id="input-marca", value=df["brand"].unique()[0])
    ], style={"width": "24%", "display": "inline-block"}),

    html.Div([
        dcc.Graph(id="grafico-volumen-vs-ganancia")
    ]),

    html.H2("Predicción de Ventas de Marca", style={"marginTop": "40px"}),
    html.Div([
        dcc.Graph(id="grafico-prediccion")
    ])
])

# Callback para gráfica 1
@app.callback(
    Output("grafico-volumen-vs-ganancia", "figure"),
    [Input("filtro-gerencia", "value"),
     Input("filtro-segmento", "value"),
     Input("input-marca", "value")]
)
def actualizar_grafico(gerencia, subcanal, marca):
    df_filtrado = df[
        (df["Gerencia"] == gerencia) &
        (df["subcanal"] == subcanal) &
        (df["brand"] == marca)
    ]

    if df_filtrado.empty:
        return px.bar(title="No hay datos disponibles para estos filtros")

    fig = px.scatter(df_filtrado, x="vol", y="desc",
                    labels={"vol": "Volumen Vendido Anterior",
                            "desc": "Descuento Anterior"},
                    title="Volumen vs. Descuento")

    return fig

# Callback para gráfica 2
@app.callback(
    Output("grafico-prediccion", "figure"),
    [Input("filtro-gerencia", "value"),
     Input("filtro-segmento", "value"),
     Input("input-marca", "value")]
)
def actualizar_prediccion(gerencia, subcanal, marca):
    df_filtrado = df[
        (df["Gerencia"] == gerencia) &
        (df["subcanal"] == subcanal) &
        (df["brand"] == marca)
    ]

    if df_filtrado.empty:
        return px.line(title="No hay datos para la predicción")

    # Agrupar por mes para reducir llamadas
    df_grouped = df_filtrado.groupby("ym").agg({
        "desc": "mean",
        "vol": "mean"
    }).reset_index()

    predictions = []
    for _, row in df_grouped.iterrows():
        payload = {
            "inputs": [{
                "desc_antes_m1": float(row["desc"]),
                "vol_vendido_antes_m1": float(row["vol"]),
                "Gerencia": gerencia,
                "subcanal": subcanal
            }]
        }

        print(f"Enviando payload: {payload}")
        try:
            response = requests.post("http://34.229.135.171:8001/api/v1/predict", json=payload, timeout=5)
            if response.status_code == 200:
                pred = response.json().get("predictions", [0])[0]
                print(f"Predicción recibida: {pred}")
            else:
                print(f"Error HTTP {response.status_code}")
                pred = 0
        except Exception as e:
            print(f"Error al llamar a la API: {e}")
            pred = 0

        predictions.append(pred)

    df_grouped["prediction"] = predictions
    df_grouped["ym"] = pd.to_datetime(df_grouped["ym"])  # asegurar formato

    fig = px.line(df_grouped, x="ym", y="prediction",
                  labels={"ym": "Mes", "prediction": "Probabilidad de Compra"},
                  title="Predicción de Compra por Mes")
    return fig

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
