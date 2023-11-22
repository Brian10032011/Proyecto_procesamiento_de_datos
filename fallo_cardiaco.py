from datasets import load_dataset
import numpy as np
import pandas as pd
import requests


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

def obtener_valor (diccionario,clave):
    return diccionario.get(clave,None)

clave_deseada = 'age' 
conversion_dataframe['age']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'has_anaemia' 
conversion_dataframe['has_anaemia']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'creatinine_phosphokinase_concentration_in_blood' 
conversion_dataframe['creatinine_phosphokinase_concentration_in_blood']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'has_diabetes' 
conversion_dataframe['has_diabetes']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'heart_ejection_fraction' 
conversion_dataframe['heart_ejection_fraction']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'has_high_blood_pressure' 
conversion_dataframe['has_high_blood_pressure']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'platelets_concentration_in_blood' 
conversion_dataframe['platelets_concentration_in_blood']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'serum_creatinine_concentration_in_blood' 
conversion_dataframe['serum_creatinine_concentration_in_blood']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'serum_sodium_concentration_in_blood' 
conversion_dataframe['serum_sodium_concentration_in_blood']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'is_male' 
conversion_dataframe['is_male']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'is_smoker' 
conversion_dataframe['is_smoker']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'days_in_study' 
conversion_dataframe['days_in_study']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

clave_deseada = 'is_dead' 
conversion_dataframe['is_dead']= conversion_dataframe['train'].apply(obtener_valor,clave=clave_deseada)

conversion_dataframe = conversion_dataframe.drop('train',axis=1)
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
PROCESANDO INFORMACION EN BRUTO
'''

def recibirUrl(url,archivo):

    response = requests.get("https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv")

    if response.status_code == 200:
        with open("datos_en_bruto.csv",'w') as archivo:
            archivo.write(response.text)
        print("Archivo creado exitosamente")
        return f"Datos descargados y guardados en '{archivo}' exitosamente."
        
    else:
        return f"Error en la solicitud. Código de estado: {response.status_code}"

if __name__ == "__main__":
    #especificando url y nombre del archivo
    url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
    archivo = "datos_descargados.csv"

    # Llamando a la función para descargar y guardar los datos
    resultado = recibirUrl(url, archivo)
    print(resultado)




'''
LIMPIEZA Y PREPARACION DE DATOS
'''
#Verificando que no existan valores faltantes
conversion_dataframe.info()

#Verificando que no existan filas repetidas
print (conversion_dataframe.duplicated())

#Verificando si existen valores atípicos y eliminarlos en la columna 'age' (0 errores atipicos)
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
# print (val_atipicos_creatinine)

#Verificando si existen valores atípicos y eliminarlos en la columna 'heart_ejection_fraction'(2 errores atipicos)
df_ordenado_heart = val_atipicos_creatinine.sort_values(by='heart_ejection_fraction', ascending=True)

Q1 = df_ordenado_heart['heart_ejection_fraction'].quantile(0.25)
Q3 = df_ordenado_heart['heart_ejection_fraction'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

val_atipicos_heart = df_ordenado_heart[(df_ordenado_heart['heart_ejection_fraction'] >= lower_bound) & (df_ordenado_heart['heart_ejection_fraction'] <= upper_bound)]
# print (val_atipicos_heart)

#Verificando si existen valores atípicos y eliminarlos en la columna 'platelets_concentration_in_blood' (21 errores atipicos)
df_ordenado_platelets = val_atipicos_heart.sort_values(by='platelets_concentration_in_blood', ascending=True)

Q1 = df_ordenado_platelets['platelets_concentration_in_blood'].quantile(0.25)
Q3 = df_ordenado_platelets['platelets_concentration_in_blood'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

val_atipicos_platelets = df_ordenado_platelets[(df_ordenado_platelets['platelets_concentration_in_blood'] >= lower_bound) & (df_ordenado_platelets['platelets_concentration_in_blood'] <= upper_bound)]
# print (val_atipicos_platelets)

#Verificando si existen valores atípicos y eliminarlos en la columna 'serum_creatinine_concentration_in_blood'(29 errores atipicos)
df_ordenado_serum = val_atipicos_platelets.sort_values(by='serum_creatinine_concentration_in_blood', ascending=True)

Q1 = df_ordenado_serum['serum_creatinine_concentration_in_blood'].quantile(0.25)
Q3 = df_ordenado_serum['serum_creatinine_concentration_in_blood'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

val_atipicos_serum = df_ordenado_serum[(df_ordenado_serum['serum_creatinine_concentration_in_blood'] >= lower_bound) & (df_ordenado_serum['serum_creatinine_concentration_in_blood'] <= upper_bound)]
# print (val_atipicos_serum)

#Verificando si existen valores atípicos y eliminarlos en la columna 'serum_sodium_concentration_in_blood' (4 errores atipicos)
df_ordenado_sodio = val_atipicos_serum.sort_values(by='serum_sodium_concentration_in_blood', ascending=True)

Q1 = df_ordenado_sodio['serum_sodium_concentration_in_blood'].quantile(0.25)
Q3 = df_ordenado_sodio['serum_sodium_concentration_in_blood'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_limpio = df_ordenado_sodio[(df_ordenado_sodio['serum_sodium_concentration_in_blood'] >= lower_bound) & (df_ordenado_sodio['serum_sodium_concentration_in_blood'] <= upper_bound)]
print (df_limpio)

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

df_limpio['Categoría de Edad'] = pd.cut(df_limpio['age'], bins=edades, labels=etiquetas, right=False)
print (df_limpio)

df_limpio.to_csv("resultado_df_limpio", index = False)


'''
ENCAPSULANDO LOGICA ANTERIOR
'''

def limpiar_y_categorizar_datos(df_limpio):
    edades = [0, 12, 19, 39, 59, float('inf')]
    etiquetas = ['Niño', 'Adolescente', 'Joven adulto', 'Adulto', 'Adulto mayor']

    df_limpio['Categoría de Edad'] = pd.cut(df_limpio['age'], bins=edades, labels=etiquetas, right=True)

    # Imprimir el DataFrame resultante después de la limpieza y categorización
    print(df_limpio)

    # Guardar el DataFrame resultante en un archivo CSV
    df_limpio.to_csv("resultado_df_limpio.csv", index=False)

    return df_limpio   

def eliminar_valores_atipicos_por_columna(df_limpio,columna):

    df_ordenado = df_limpio.sort_values(by=columna,ascending=True)
    Q1 = df_ordenado[columna].quantile(0.25)
    Q3 = df_ordenado[columna].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_limpio = df_ordenado[(df_ordenado[columna] >= lower_bound) & (df_ordenado[columna] <= upper_bound)]

    return df_limpio

if __name__ == '__main__':
       
    df_limpio = eliminar_valores_atipicos_por_columna(conversion_dataframe, 'heart_ejection_fraction')
    print("DataFrame limpio (heart_ejection_fraction):")
    print(df_limpio)
    df_resultante = limpiar_y_categorizar_datos(df_limpio)
    # print(df_resultante)
