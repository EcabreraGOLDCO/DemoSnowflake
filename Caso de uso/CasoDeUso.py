# Importar librerías necesarias
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from Levenshtein import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Configuración de la conexión a Snowflake
connection_parameters = {
    "account": "ej17609.ca-central-1.aws",
    "user": "Jairc35",
    "password": "Fenix1103.",
    "role": "ACCOUNTADMIN",
    "warehouse": "COMPUTE_WH",
    "database": "productos_test",
    "schema": "PUBLIC",
}

# Crear sesión con Snowflake
session = Session.builder.configs(connection_parameters).create()

# Función para corregir typos basados en la moda
def corregir_typos(marca, moda):
    """Corrige typos en la columna 'marca' si la distancia de Levenshtein es menor a 3."""
    return moda if distance(marca, moda) < 3 else marca

# Obtener la tabla de productos desde Snowflake
df = session.table("productos")

# Calcular la moda de la columna 'marca'
moda = (
    df.group_by(col("marca"))
    .agg({"marca": "count"})
    .order_by(col("count").desc())
    .first()[0]  # Obtener la marca más frecuente
)

# Aplicar la corrección de typos
df = df.with_column("marca_corregida", col("marca").apply(lambda marca: corregir_typos(marca, moda)))

# Guardar la tabla corregida en Snowflake
df.write.mode("overwrite").save_as_table("productos_corregidos")

# Convertir la tabla a un DataFrame de Pandas para trabajar con TF-IDF
df_pandas = df.to_pandas()

# Crear vectores TF-IDF para 'nombre' y 'descripcion'
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_pandas["nombre"] + " " + df_pandas["descripcion"])

# Calcular la matriz de similitud coseno entre productos
similarity_matrix = cosine_similarity(tfidf_matrix)

# Función para inferir el valor faltante en la columna 'tipo'
def inferir_tipo(similarity_matrix, producto_index, df_pandas):
    """Rellena el campo 'tipo' usando productos similares basados en la matriz de similitud."""
    similares = similarity_matrix[producto_index].argsort()[-5:][::-1]  # Top 5 productos similares
    tipos_similares = df_pandas.iloc[similares]["tipo"].dropna()  # Filtrar valores no nulos
    return tipos_similares.mode().iloc[0] if not tipos_similares.empty else None

# Aplicar el relleno de la columna 'tipo'
df_pandas["tipo_rellenado"] = [
    inferir_tipo(similarity_matrix, idx, df_pandas)
    if pd.isna(row["tipo"]) else row["tipo"]
    for idx, row in df_pandas.iterrows()
]

# Subir los datos enriquecidos a Snowflake
df_enriched = session.create_dataframe(df_pandas)
df_enriched.write.mode("overwrite").save_as_table("productos_enriquecidos")

# Cerrar la sesión de Snowflake
session.close()