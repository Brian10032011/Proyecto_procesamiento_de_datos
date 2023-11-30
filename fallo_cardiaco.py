from datasets import load_dataset
import numpy as np
import pandas as pd
import requests

url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
dataset = load_dataset("mstz/heart_failure")
data = np.array (dataset["train"])

'''
Promedio edades de las personas participantes
'''
# suma = 0
# conteo = 0
# for item in data:
#     suma += item ['age']
#     conteo += 1
# print (suma/conteo)


'''
PROCESANDO INFORMACION EN BRUTO
'''

def recibirUrl(url,nombre_archivo):

    response = requests.get(url)

    if response.status_code == 200:
        with open(nombre_archivo,'w') as archivo:
            archivo.write(response.text)
        print("Archivo creado exitosamente")
        return f"Datos descargados y guardados en '{archivo}' exitosamente."
        
    else:
        return f"Error en la solicitud. Código de estado: {response.status_code}"
recibirUrl(url,"data.csv")
dataframe = pd.read_csv("data.csv")

'''
Convirtiendo estructura a DataFrame
'''

conversion_dataframe = pd.DataFrame(dataset)
print(conversion_dataframe)

'''
Division de DataFrame segun condicion is_dead
'''
condicion1 = conversion_dataframe['train'].apply(lambda x : x ['is_dead']==1)
pacientes_is_dead = conversion_dataframe[condicion1]
print (pacientes_is_dead)


condicion2 = conversion_dataframe['train'].apply(lambda x : x ['is_dead']==0)
pacientes_not_is_dead = conversion_dataframe[condicion2]
print (pacientes_not_is_dead)


'''
CALCULO PROMEDIO EDADES
'''
promedio_edades1 = pacientes_is_dead['train'].apply(lambda x : x ['age']).mean()
print (promedio_edades1)

promedio_edades2 = pacientes_not_is_dead['train'].apply(lambda x : x ['age']).mean()
print (promedio_edades2)


'''
DIVISION DATAFRAME A 299 FILAS Y 13 COLUMNAS
'''

def obtener_valores(conversion_dataframe):
    def obtener_valor(fila, columna):
        return fila.get(columna, None)

    columnas_deseadas = ['age', 'has_anaemia', 'creatinine_phosphokinase_concentration_in_blood',
                       'has_diabetes', 'heart_ejection_fraction', 'has_high_blood_pressure',
                       'platelets_concentration_in_blood', 'serum_creatinine_concentration_in_blood',
                       'serum_sodium_concentration_in_blood', 'is_male', 'is_smoker',
                       'days_in_study', 'is_dead']

    for columna in columnas_deseadas:
        conversion_dataframe[columna] = conversion_dataframe['train'].apply(lambda x: obtener_valor(x, columna))

    conversion_dataframe = conversion_dataframe.drop('train', axis=1)
    return conversion_dataframe

# Uso de la función
conversion_dataframe = obtener_valores(conversion_dataframe)
print(conversion_dataframe)

'''
Consultando tipo de datos correctos
'''
print(conversion_dataframe.dtypes)


'''
CALCULO HOMBRES VS MUJERES FUMADORAS
'''

hfumadores_vs_mfumadoras = conversion_dataframe.groupby(['is_male', 'is_smoker']).size().unstack()
print("\nresultados:")
print(hfumadores_vs_mfumadoras)


'''
LIMPIEZA Y PREPARACION DE DATOS
'''
#Verificando que no existan valores faltantes
conversion_dataframe.info()

#Verificando que no existan filas repetidas
print (conversion_dataframe.duplicated())

#Verificando si existen valores atípicos y eliminarlos en la columna 'age' (0 errores atipicos)
def limpiarPrepararDatos(dataframe):
    conversion_dataframe = dataframe
    df_ordenado_age = conversion_dataframe.sort_values(by='age', ascending=True)

    Q1 = df_ordenado_age['age'].quantile(0.25)
    Q3 = df_ordenado_age['age'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    val_atipicos_age = df_ordenado_age[(df_ordenado_age['age'] >= lower_bound) & (df_ordenado_age['age'] <= upper_bound)]
    # print (val_atipicos_age)

    #Verificando si existen valores atípicos y eliminarlos en la columna 'creatinine_phosphokinase_concentration_in_blood'(29 errores atipicos)
    df_ordenado_creatinine = conversion_dataframe.sort_values(by='creatinine_phosphokinase_concentration_in_blood', ascending=True)

    Q1 = df_ordenado_creatinine['creatinine_phosphokinase_concentration_in_blood'].quantile(0.25)
    Q3 = df_ordenado_creatinine['creatinine_phosphokinase_concentration_in_blood'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    val_atipicos_creatinine = df_ordenado_creatinine[(df_ordenado_creatinine['creatinine_phosphokinase_concentration_in_blood'] >= lower_bound) & (df_ordenado_creatinine['creatinine_phosphokinase_concentration_in_blood'] <= upper_bound)]
    # print ("Errores atipicos de creatinine :",val_atipicos_creatinine)

    #Verificando si existen valores atípicos y eliminarlos en la columna 'heart_ejection_fraction'(2 errores atipicos)
    df_ordenado_heart = val_atipicos_creatinine.sort_values(by='heart_ejection_fraction', ascending=True)

    Q1 = df_ordenado_heart['heart_ejection_fraction'].quantile(0.25)
    Q3 = df_ordenado_heart['heart_ejection_fraction'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    val_atipicos_heart = df_ordenado_heart[(df_ordenado_heart['heart_ejection_fraction'] >= lower_bound) & (df_ordenado_heart['heart_ejection_fraction'] <= upper_bound)]
    # print ("Errores atipicos de heart :",val_atipicos_heart)

    #Verificando si existen valores atípicos y eliminarlos en la columna 'platelets_concentration_in_blood' (18 errores atipicos)
    df_ordenado_platelets = val_atipicos_heart.sort_values(by='platelets_concentration_in_blood', ascending=True)

    Q1 = df_ordenado_platelets['platelets_concentration_in_blood'].quantile(0.25)
    Q3 = df_ordenado_platelets['platelets_concentration_in_blood'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    val_atipicos_platelets = df_ordenado_platelets[(df_ordenado_platelets['platelets_concentration_in_blood'] >= lower_bound) & (df_ordenado_platelets['platelets_concentration_in_blood'] <= upper_bound)]
    # print ("Errores atipicos de platelets :",val_atipicos_platelets)

    #Verificando si existen valores atípicos y eliminarlos en la columna 'serum_creatinine_concentration_in_blood'(23 errores atipicos)
    df_ordenado_serum = val_atipicos_platelets.sort_values(by='serum_creatinine_concentration_in_blood', ascending=True)

    Q1 = df_ordenado_serum['serum_creatinine_concentration_in_blood'].quantile(0.25)
    Q3 = df_ordenado_serum['serum_creatinine_concentration_in_blood'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    val_atipicos_serum = df_ordenado_serum[(df_ordenado_serum['serum_creatinine_concentration_in_blood'] >= lower_bound) & (df_ordenado_serum['serum_creatinine_concentration_in_blood'] <= upper_bound)]
    # print ("Errores atipicos de serum :",val_atipicos_serum)

    #Verificando si existen valores atípicos y eliminarlos en la columna 'serum_sodium_concentration_in_blood' (3 errores atipicos)
    df_ordenado_sodio = val_atipicos_serum.sort_values(by='serum_sodium_concentration_in_blood', ascending=True)

    Q1 = df_ordenado_sodio['serum_sodium_concentration_in_blood'].quantile(0.25)
    Q3 = df_ordenado_sodio['serum_sodium_concentration_in_blood'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_limpio = df_ordenado_sodio[(df_ordenado_sodio['serum_sodium_concentration_in_blood'] >= lower_bound) & (df_ordenado_sodio['serum_sodium_concentration_in_blood'] <= upper_bound)]
    print ("dataframe limpio :",df_limpio)

    #Verificando si existen valores atípicos y eliminarlos en la columna 'days_in_study' (0 errores atipicos)
    df_ordenado_dias_en_estudio = conversion_dataframe.sort_values(by='days_in_study', ascending=True)

    Q1 = df_ordenado_dias_en_estudio['days_in_study'].quantile(0.25)
    Q3 = df_ordenado_dias_en_estudio['days_in_study'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    val_atipicos_dias_en_estudio = df_ordenado_dias_en_estudio[(df_ordenado_dias_en_estudio['days_in_study'] >= lower_bound) & (df_ordenado_dias_en_estudio['days_in_study'] <= upper_bound)]
    # print (val_atipicos_dias_en_estudio)
    '''
    CREACION DE COLUMNA CATEGORIZANDO POR EDADES
    '''
    edades = [0, 12,19, 39, 59, float('inf')]

    etiquetas = ['Niño', 'Adolescente', 'Jovenes adulto', 'Adulto', 'Adulto mayor']
    #import pdb;pdb.set_trace()
    df_limpio = df_limpio.copy()
    df_limpio['Categoría de Edad'] = pd.cut(df_limpio['age'], bins=edades, labels=etiquetas, right=False)
    print (df_limpio)

    df_limpio.to_csv("resultado_df_limpio.csv", index = False)
limpiarPrepararDatos(conversion_dataframe)

 