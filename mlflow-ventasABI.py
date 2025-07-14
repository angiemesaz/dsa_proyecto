# Importar librerías
pip install sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from mlflow.models import infer_signature

# Cargar el conjunto de datos
clientes = pd.read_excel('clientes_final.xlsx')
ventas = pd.read_excel('venta_final.xlsx')

# Procesar conjunto de datos
ventas.drop_duplicates(inplace=True)
clientes.drop_duplicates(inplace=True)

abt = pd.merge(ventas, clientes, on='Cliente', how='left')
abt_final = abt[['Gerencia', 'subcanal', 'desc', 'brand', 'ym', 'Cliente', 'vol']]
abt_final['target'] = np.where(abt['brand'] == 'Marca1', 1, 0)

# Creación de variable de volumen vendido anteriormente al cliente de esta misma marca
abt_final['ym'] = pd.to_datetime(abt_final['ym'], format='%Y%m')
df_marca1 = abt_final[abt_final['brand'] == 'Marca1'].copy()
df_marca1 = df_marca1.sort_values(by=['Cliente', 'ym'])
df_marca1['vol_vendido_antes_m1'] = df_marca1.groupby('Cliente')['vol'].cumsum() - df_marca1['vol']
result = abt_final.merge(df_marca1[['Cliente', 'ym', 'vol_vendido_antes_m1']],
                         on=['Cliente', 'ym'],
                         how='left')
result['vol_vendido_antes_m1'] = result['vol_vendido_antes_m1'].fillna(0)

# Creación del descuento dado al cliente en los periodos anteriores
df_desc = result.loc[result['brand'] == 'Marca1']
df_desc['abs_desc'] = abs(df_desc['desc'])
df_desc = df_desc.sort_values(by=['Cliente', 'ym'])
df_desc['desc_antes_m1'] = df_desc.groupby('Cliente')['abs_desc'].cumsum() - df_desc['abs_desc']
result2 = result.merge(df_desc[['Cliente', 'ym', 'desc_antes_m1']],
                         on=['Cliente', 'ym'],
                         how='left')
result2['desc_antes_m1'] = result2['desc_antes_m1'].fillna(0)

abt_modelo = result2.groupby(['Cliente', 'ym', 'Gerencia', 'subcanal']).agg({'desc_antes_m1': 'max',
                                        'vol_vendido_antes_m1' : 'max',
                                        'target' : 'max'}).reset_index(drop=False).drop(columns=['Cliente',
                                                                                                 'ym'])

X = abt_modelo[['desc_antes_m1', 'vol_vendido_antes_m1', 'Gerencia', 'subcanal']]
y = abt_modelo['target']

numeric_features = ['desc_antes_m1', 'vol_vendido_antes_m1']
categorical_features = ['Gerencia', 'subcanal']

numeric_transformer = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'
)
X_processed = preprocessor.fit_transform(X[numeric_features])
X_categorical = X[categorical_features]
X_final = np.hstack((X_processed, X_categorical))

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=0)

# Configurar el experimento en MLflow
experiment = mlflow.set_experiment("sklearn-ventasABI")

# Iniciar el registro del experimento
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Parámetros del modelo
    cat_features = [2,3]
    iterations = None 
    early_stopping_rounds=None
    learning_rate = None
    depth = None

    # Crear y entrenar el modelo
    model = CatBoostClassifier(cat_features = cat_features,
                              iterations = iterations, 
                              early_stopping_rounds = early_stopping_rounds,
                              learning_rate = learning_rate,
                              depth = depth)
    model.fit(X_train, y_train, cat_features = cat_features,
              verbose=200)

    # Realizar predicciones
    predictions = model.predict(X_test)

    # Registrar parámetros del modelo
    mlflow.log_param("iterations", iterations)
    mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("depth", depth)

    # Calcular métricas
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)

    # Registrar métricas
    mlflow.log_metric("accuracy_score", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall_score", recall)
    mlflow.log_metric("precision_score", precision)
    mlflow.log_metric("roc_auc_score", roc_auc)

    # Crear ejemplo de entrada y firma del modelo
    input_example = pd.DataFrame(X_train[:1], columns=X.columns)
    signature = infer_signature(X_train, model.predict(X_train))

    # Registrar el modelo con ejemplo de entrada y firma
    mlflow.sklearn.log_model(
        model,
        name="catboost-model",
        input_example=input_example,
        signature=signature
    )

    # Imprimir métricas
    print(f"Accuracy: {accuracy:.2f}")
    print(f"f1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC auc: {roc_auc:.4f}")