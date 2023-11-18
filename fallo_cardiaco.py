from datasets import load_dataset
import numpy as np
import pandas as pd

dataset = load_dataset("mstz/heart_failure")
data = np.array (dataset["train"])

'''
Promedio edades de las personas participantes
'''
suma = 0
conteo = 0
for item in data:
    suma += item ['age']
    conteo += 1
print (suma/conteo)


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


