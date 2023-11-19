from datasets import load_dataset
import numpy as np
import pandas as pd

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
Clacular promedios de edades de cada DataFrame
'''
promedio_edades1 = pacientes_is_dead['train'].apply(lambda x : x ['age']).mean()
print (promedio_edades1)

promedio_edades2 = pacientes_not_is_dead['train'].apply(lambda x : x ['age']).mean()
print (promedio_edades2)

'''
Dividiendo el Dataframe segun cada caracteristica [299*13]
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
Calculo de hombres fumadores vs mujeres fumadoras
'''

hfumadores_vs_mfumadoras = conversion_dataframe.groupby(['is_male', 'is_smoker']).size().unstack()
print("\nresultados:")
print(hfumadores_vs_mfumadoras)



