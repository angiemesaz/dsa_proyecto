
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# Cargar datos y modelo
df = pd.read_csv("abt.csv")
df['ym'] = pd.to_datetime(df['ym'])
df['ym_str'] = df['ym'].dt.strftime('%Y-%m')

model = joblib.load("model.pkl")

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

    fig = px.scatter(df_filtrado, x="vol_vendido_antes_m1", y="desc_antes_m1", color="target",
                     labels={"vol_vendido_antes_m1": "Volumen Vendido Anterior",
                             "desc_antes_m1": "Descuento Anterior"},
                     title="Volumen vs. Descuento - Clasificado por Compra")
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

    # Crear conjunto de entrada
    X = df_filtrado[["desc_antes_m1", "vol_vendido_antes_m1", "Gerencia", "subcanal"]].copy()
    X[["desc_antes_m1", "vol_vendido_antes_m1"]] = X[["desc_antes_m1", "vol_vendido_antes_m1"]].astype(float)
    X["Gerencia"] = X["Gerencia"].astype(str)
    X["subcanal"] = X["subcanal"].astype(str)

    numeric_features = ["desc_antes_m1", "vol_vendido_antes_m1"]
    categorical_features = ["Gerencia", "subcanal"]
    numeric_transformer = MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="passthrough"
    )

    X_processed = preprocessor.fit_transform(X[numeric_features])
    X_categorical = X[categorical_features].values
    X_final = np.hstack((X_processed, X_categorical))

    y_pred = model.predict(X_final)
    df_filtrado["prediction"] = y_pred

    df_pred = df_filtrado.groupby("ym")["prediction"].mean().reset_index()
    fig = px.line(df_pred, x="ym", y="prediction",
                  labels={"ym": "Mes", "prediction": "Probabilidad de Compra"},
                  title="Predicción de Compra por Mes")
    return fig

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
