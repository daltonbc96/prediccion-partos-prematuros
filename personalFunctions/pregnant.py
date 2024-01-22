def process_data(raw_data, json_dictionary, check_columns_in_data_not_in_json=True, check_columns_in_json_not_in_data=True):
    import numpy as np
    import pandas as pd
    import ast
    import re
    from tqdm import tqdm
    """
    Procesa el DataFrame 'raw_data' basado en la metadata proporcionada en 'json_dictionary'.
    Opcionalmente verifica las columnas presentes en el DataFrame pero no en json_dictionary y viceversa.
    
    :param json_dictionary: Objeto JSON con metadatos que definen los tipos de las columnas.
    :param raw_data: DataFrame de Pandas que se procesará.
    :param check_columns_in_data_not_in_json: Booleano para activar la verificación de columnas en DATA pero no en JSON.
    :param check_columns_in_json_not_in_data: Booleano para activar la verificación de columnas en JSON pero no en DATA.
    :return: DataFrame procesado, lista de columnas en DATA pero no en JSON, lista de columnas en JSON pero no en DATA.
    """
    columns_in_data_not_in_json = []
    columns_in_json_not_in_data = []

    # Verificar las columnas que están en el DataFrame pero no en json_dictionary
    if check_columns_in_data_not_in_json:
        for column in raw_data.columns:
            if column not in json_dictionary:
                columns_in_data_not_in_json.append(column)

       
    # Procesar cada columna basada en su tipo en json_dictionary
    for column_name, metadata in tqdm(json_dictionary.items()):
        if check_columns_in_json_not_in_data and column_name not in raw_data.columns:
            columns_in_json_not_in_data.append(column_name)
            continue
            
        column_type = metadata.get('type')

        # Definir funciones de procesamiento específicas para cada tipo
        def process_time(value) :
            if pd.isna(value) or value == '':
                return np.nan
            try:
                return pd.to_datetime(value, dayfirst=True, errors='coerce')
            except (ValueError, pd.errors.ParserError):
                return np.nan

        def process_numeric(value):
            if pd.isna(value) or value == '':
                return np.nan
            return float(value)

        def process_text(value):
            if pd.isna(value) or value == '':
                return np.nan
            value = re.sub(r'[^\w\s]', '', str(value)).lower()
            return value

        def process_boolean(value):
            if pd.isna(value) or value == '':
                return np.nan   
            numeric_value = float(value)
            return bool(numeric_value)


        def process_code(value):
            if pd.isna(value) or value == '':
                return np.nan
            return str(value)

        def process_date(value):
            if pd.isna(value) or value == '':
                return np.nan
            try:
                return pd.to_datetime(value, dayfirst=True, errors='coerce')
            except ValueError:
                return np.nan
    

        def process_institution(value):
            if pd.isna(value) or value == '':
                return pd.Series(dtype='object')  # Especifica o tipo de dados como 'object'
            institution_dict = ast.literal_eval(value)
            return pd.Series(institution_dict) if isinstance(institution_dict, dict) else pd.Series(dtype='object')
        

        def process_enumeration(value):
            if pd.isna(value) or (isinstance(value, str) and not value.isnumeric()):
                return np.nan
            value = str(int(float(value)))
            spanish_labels = metadata['properties']['names'].get('spanish', {})
            return spanish_labels.get(value, f"No hay etiqueta para el valor {value}")

        # Aplicar la función correspondiente y ajustar el tipo de columna si es necesario
        if column_type == 'TIME':
            raw_data[column_name] = raw_data[column_name].apply(process_time)
        elif column_type == 'NUMERIC':
            raw_data[column_name] = raw_data[column_name].apply(process_numeric)
        elif column_type in ['TEXT', 'LONGTEXT']:
            raw_data[column_name] = raw_data[column_name].apply(process_text).astype('string')
        elif column_type == 'BOOLEAN':
            raw_data[column_name] = raw_data[column_name].apply(process_boolean).astype('boolean')
        elif column_type == 'CODE':
            raw_data[column_name] = raw_data[column_name].apply(process_code).astype('category')
        elif column_type == 'DATE':
            raw_data[column_name] = raw_data[column_name].apply(process_date)
        elif column_type == 'INSTITUTION':
            expanded_cols = raw_data[column_name].apply(process_institution).apply(pd.Series)
            for col in expanded_cols.columns:
                new_col_name = f"{column_name}_{col}"
                raw_data[new_col_name] = expanded_cols[col].astype('string')
            raw_data.drop(column_name, axis=1, inplace=True)
        elif column_type == 'ENUMERATION':
            raw_data[column_name] = raw_data[column_name].apply(process_enumeration).astype('category')

    return raw_data, columns_in_data_not_in_json, columns_in_json_not_in_data

def get_types(data):
    """
   Analiza JSON para descubrir los diferentes tipos de datos que existen.
    """
    types = set() 
    
    for item in data.values():
        if 'type' in item:
            types.add(item['type'])

    return types


def dummify_categorical_columns(df, ignore_cols=None):
    """
    Convierte las columnas categóricas de un DataFrame en variables dummy/indicadoras,
    excepto las columnas especificadas en ignore_cols, manteniendo las demás columnas del DataFrame.

    Esta función es útil para preparar los datos antes de aplicar algoritmos de aprendizaje
    automático, especialmente aquellos que requieren entrada numérica.

    :param df: DataFrame de Pandas que se procesará.
    :param ignore_cols: Lista de nombres de columnas categóricas a ser ignoradas en el proceso de dumificación.
    :return: Un nuevo DataFrame con las columnas categóricas convertidas en variables dummy.
    
    Uso:
    df_dummies = dummify_categorical_columns(df_original, ignore_cols=['col1', 'col2'])
    """

    import pandas as pd

    # Identificar las columnas categóricas
    categorical_cols = df.select_dtypes(include=['category']).columns

    # Excluir las columnas que se quieren ignorar
    if ignore_cols is not None:
        categorical_cols = [col for col in categorical_cols if col not in ignore_cols]

    # Crear variables dummy para las columnas categóricas
    dummies = pd.get_dummies(df, columns=categorical_cols)

    return dummies




def adapt_column_name(name):
    import unidecode
    """
    Adapta el nombre de una columna para hacerlo más adecuado para el análisis de datos.

    Esta función realiza las siguientes transformaciones:
    - Elimina acentos y caracteres especiales.
    - Reemplaza espacios con guiones bajos (_).
    - Convierte el nombre a minúsculas.
    - Elimina todos los caracteres no alfanuméricos excepto guiones bajos.

    :param name: El nombre original de la columna.
    :return: El nombre de la columna adaptado.
    """
    name = unidecode.unidecode(name)
    name = name.replace(" ", "_").lower()
    name = ''.join(e for e in name if e.isalnum() or e == '_')
    return name

def rename_columns(df, column_mapping):
    """
    Renombra las columnas de un DataFrame basándose en un mapeo proporcionado y
    añade el nombre original seguido de un guión bajo y el nombre en ['names']['spanish'].

    :param df: DataFrame cuyas columnas serán renombradas.
    :param column_mapping: Diccionario con mapeo de nombres de columnas.
    :return: DataFrame con columnas renombradas.
    """
    # Mapeando todas las columnas del DataFrame
    new_column_names = {col: adapt_column_name(col) for col in df.columns}

    # Actualizando el mapeamiento con las columnas especificadas en el JSON
    for key, value in column_mapping.items():
        if key in new_column_names:
            new_name = adapt_column_name(value['names']['spanish'])
            new_column_names[key] = f"{new_column_names[key]}_{new_name}"

    # Renombrando las columnas en el DataFrame
    df = df.rename(columns=new_column_names)
    return df

def missing_columns(df, missing_threshold=50):
    """
    Identifica las columnas en un DataFrame que tienen un alto porcentaje de valores faltantes 
    o que contienen un único valor.

    Esta función es útil para la limpieza de datos y la preparación previa al análisis, 
    permitiendo identificar columnas que pueden no ser útiles para el análisis debido a 
    la falta de variabilidad o a la presencia excesiva de valores faltantes.

    :param df: DataFrame de Pandas para analizar.
    :param missing_threshold: Umbral para considerar una columna con valores faltantes 
                              altos en porcentaje (%).
    :return: Tres listas:
             - cols_high_missing: Columnas con un porcentaje de valores faltantes superior al umbral.
             - cols_single_value: Columnas con un único valor.
             - combined_cols: Combinación de ambas listas anteriores.

    Uso:
    cols_high_missing, cols_single_value, combined_cols = missing_columns(df, 50)
    """
    # Calcular el porcentaje de valores faltantes por columna
    missing_percent = df.isnull().mean() * 100

    # Identificar columnas con un alto porcentaje de valores faltantes
    cols_high_missing = missing_percent[missing_percent > missing_threshold].index.tolist()

    # Identificar columnas con un único valor
    cols_single_value = [col for col in df.columns if len(df[col].unique()) == 1]

    # Combinar las listas de columnas
    combined_cols = list(set(cols_high_missing + cols_single_value))

    return cols_high_missing, cols_single_value, combined_cols

def plot_missing_data_and_unique_value(df, sort_descending=False, missing_threshold=100, figure_height=400):
    """
    Genera dos gráficos de barras utilizando Plotly Express. El primer gráfico muestra el porcentaje de datos faltantes en cada columna del DataFrame 'df'. El segundo gráfico muestra las columnas que contienen solo un valor único (excluyendo NaN).

    :param df: DataFrame de pandas que se va a analizar.
    :param sort_descending: Booleano. Si es True, ordena los datos de forma descendente basado en el porcentaje de datos faltantes.
    :param missing_threshold: Umbral para considerar una columna como completamente faltante. Por defecto es 100, lo que indica un 100% de datos faltantes.
    :param figure_height: Altura de los gráficos de barras generados. Por defecto es 400.

    :return: Una tupla de dos figuras de Plotly Express. La primera figura representa el porcentaje de datos faltantes por columna. La segunda figura muestra el porcentaje de columnas con un único valor.

    La función calcula primero el total de columnas en el DataFrame. Luego, calcula el porcentaje de datos faltantes en cada columna y cuenta las columnas con un 100% de datos faltantes, basado en el 'missing_threshold'. También identifica las columnas que tienen solo un valor único (excluyendo NaN) y cuenta cuántas hay. Los resultados se presentan en gráficos de barras utilizando Plotly Express.
    """

    import pandas as pd
    import plotly.express as px
    
    
    total_columns = len(df.columns)

   
    missing_data = (df.isnull().sum() / len(df)) * 100

   
    total_missing_100_percent = (missing_data >= missing_threshold).sum()

   
    unique_value_cols = df.nunique(dropna=True)
    unique_value_percentage = (unique_value_cols == 1).astype(int) * 100

    
    total_unique_values = (unique_value_cols == 1).sum()

   
    if sort_descending:
        missing_data = missing_data.sort_values(ascending=False)
        unique_value_percentage = unique_value_percentage[missing_data.index]

   
    fig_missing = px.bar(missing_data, x=missing_data.index, y=missing_data.values,
                         labels={'x': 'Column', 'y': 'Missing Data (%)'},
                         title=f'Percentage of Missing Data by Column (Number of Columns with {missing_threshold}% Missing: {total_missing_100_percent}) - Total Columns: {total_columns}',
                         height=figure_height)

    fig_unique = px.bar(unique_value_percentage, x=unique_value_percentage.index, y=unique_value_percentage.values,
                        labels={'x': 'Column', 'y': 'Unique Value Presence (%)'},
                        title=f'Columns with Only One Unique Value (Total: {total_unique_values} - Total Columns: {total_columns}',
                        height=figure_height)

    return fig_missing, fig_unique


def plot_corr_graph(correlation_matrix, labels, target_variable, threshold=0.0, target_position = -2.0):
    """
    Genera un gráfico de red a partir de una matriz de correlación, mostrando las relaciones 
    entre las variables que superan un umbral de correlación especificado.

    Esta función es útil para visualizar cómo las diferentes variables están correlacionadas 
    entre sí y, en particular, cómo se relacionan con una variable objetivo.

    :param correlation_matrix: Matriz de correlación. Puede ser un ndarray de NumPy o un DataFrame de Pandas.
    :param labels: Lista de etiquetas para las variables en la matriz de correlación.
    :param target_variable: Variable objetivo que se desea resaltar en el gráfico.
    :param threshold: Umbral de correlación para determinar si se incluye una relación en el gráfico.
                      Por defecto es 0.0, lo que significa que todas las correlaciones se incluyen.

    :return: Objeto de figura de Plotly con el gráfico de red generado.

    Nota: La función se encarga de limitar las etiquetas a los primeros 4 caracteres para
    mantener el gráfico legible. Las relaciones se muestran como líneas entre los nodos, donde
    el color y el grosor de la línea indican la fuerza y el tipo de correlación (positiva o negativa).

    Uso:
    fig = plot_corr_graph(correlation_matrix, labels, 'variable_objetivo')
    fig.show()  # Muestra el gráfico en un navegador o en un entorno interactivo como Jupyter.
    """
    import plotly.graph_objs as go
    import networkx as nx
    import math
    import numpy as np

    
    if not isinstance(correlation_matrix, np.ndarray):
       correlation_matrix = correlation_matrix.to_numpy()

   
    labels = [label[:4] for label in labels]


    G = nx.Graph()


    for i, label in enumerate(labels):
        G.add_node(label)


    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            weight = correlation_matrix[i, j]
            if abs(weight) >= threshold:
                G.add_edge(labels[i], labels[j], weight=weight)


    target_label = target_variable[:4]
    for node in list(G.nodes):
        if node != target_label and all(abs(correlation_matrix[labels.index(node), labels.index(other_node)]) < threshold for other_node in labels if other_node != node):
            G.remove_node(node)

 
    pos = nx.spring_layout(G, k=0.3, iterations=50)


    pos[target_label] = (target_position, 0)


    traces = []
    edge_annotations = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        color = 'red' if weight < 0 else 'blue'
        
        traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', 
                                 line=dict(color=color, width=2)))
        
 
        if weight not in [1, -1]:
            weight_text = str(round(weight, 2))
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            edge_annotations.append(
                dict(
                    x=mid_x, 
                    y=mid_y, 
                    xref='x', yref='y',
                    text=weight_text, 
                    showarrow=False, 
                    font=dict(color='white', size=10),
                    bgcolor=color,  
                    bordercolor='black',  
                    borderwidth=1,
                    borderpad=4
                )
            )

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = list(G.nodes())
    node_color = ['green' if node == target_variable[:4] else 'white' for node in G.nodes()]

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                            textposition='middle center', hoverinfo='text', textfont_size=12,
                            marker=dict(size=50, color=node_color, line=dict(width=2)))

    fig = go.Figure(data=traces + [node_trace], layout=go.Layout(showlegend=False, 
                                                                 margin=dict(t=20, b=20, l=20, r=20),
                                                                 width=800, height=600,
                                                                 xaxis_zeroline=False, yaxis_zeroline=False,
                                                                 xaxis_showgrid=False, yaxis_showgrid=False,
                                                                 plot_bgcolor='rgba(0,0,0,0)',
                                                                 annotations=edge_annotations))
    return fig

def identificar_columnas_no_numericas(df):
    import pandas as pd
    """
    Identifica las columnas en un DataFrame de pandas que son de tipo
    fecha, hora o cadena (string).

    Parámetros:
    df (pandas.DataFrame): El DataFrame sobre el cual realizar la identificación.

    Devuelve:
    list: Una lista con los nombres de las columnas que son de tipo fecha, hora o cadena.

    Ejemplo:
    >>> df = pd.DataFrame({'fecha': pd.to_datetime(['2023-01-01']), 'numero': [1], 'texto': ['hola']})
    >>> identificar_columnas(df)
    ['fecha', 'texto']
    """

    columnas_fecha_hora_string = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            columnas_fecha_hora_string.append(col)

    return columnas_fecha_hora_string



def matriz_correlacion_optimizada(df, variable_especifica, min_correlacion=0, ignorar_columnas=[]):
    """
    Genera una matriz de correlación optimizada, excluyendo las columnas especificadas y luego determinando
    las variables relacionadas con la variable específica con una correlación (en valor absoluto) igual o
    superior al valor mínimo suministrado. Posteriormente, calcula la matriz de correlación solo entre las
    variables seleccionadas.

    Parámetros:
    df (DataFrame): DataFrame de Pandas con los datos.
    variable_especifica (str): Nombre de la variable específica para la correlación.
    min_correlacion (float): Valor mínimo de correlación (en valor absoluto) para considerar una variable.
    ignorar_columnas (list): Lista de nombres de columnas a excluir del cálculo de correlación.

    Retorna:
    DataFrame: Matriz de correlación optimizada.
    """

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    if variable_especifica not in df.columns:
        raise ValueError(f"La variable '{variable_especifica}' no se encuentra en el DataFrame.")

    # Excluyendo las columnas especificadas
    df_reducido = df.drop(columns=ignorar_columnas)

    # Calcula la correlación de la variable específica con todas las demás en el DataFrame reducido
    correlaciones = df_reducido.corrwith(df_reducido[variable_especifica]).abs()

    # Filtra las variables que cumplen con el umbral de correlación
    variables_relevantes = correlaciones[correlaciones >= min_correlacion].index

    # Calcula la matriz de correlación solo entre las variables filtradas
    matriz_corr_optimizada = df_reducido[variables_relevantes].corr()

    print(f"Número de variables con una correlación igual o superior al límite con la variable objetivo: {matriz_corr_optimizada.shape[1]} - Correlaciones con filtro >={min_correlacion}")

    return matriz_corr_optimizada


def graficar_matriz_correlacion(corr_matrix):

    """
    Grafica una matriz de correlación usando seaborn.

    Parámetros:
    corr_matrix (pandas.DataFrame): Matriz de correlación a graficar.
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Crear una máscara para ocultar la parte superior derecha del gráfico
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Configurar el tamaño del gráfico
    f, ax = plt.subplots(figsize=(11, 9))

    # Generar una paleta de colores personalizada
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Dibujar el mapa de calor con la máscara y la proporción correcta
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, square=True)
    plt.show()