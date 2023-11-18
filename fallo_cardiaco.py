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

