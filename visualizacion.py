import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
LEER Y CONVERTIR A DATAFRAME EL ARCHIVO LIMPIO
'''
df = pd.read_csv("resultado_df_limpio.csv")
dataframe = pd.DataFrame(df)
print(dataframe)

'''
1.GRAFICANDO LA DISTRIBUCION DE LAS EDADES EN UN HISTOGRAMA  
'''
edades = dataframe['age'].unique()
len(edades)

plt.hist(edades, bins = 10 , edgecolor = 'black')
plt.title('Distribución de Edades')
plt.xlabel('Edades')
plt.ylabel('Cantidad de personas')
plt.show()

'''
2. GRAFICA DE HISTOGRAMA AGRUPADO POR HOMBRE Y MUJER
'''
hombres_anemi = dataframe.query('is_male == True and has_anaemia == True').value_counts().shape[0]
mujeres_anemi = dataframe.query('is_male == False and has_anaemia == True').value_counts().shape[0]
hombres_diabe = dataframe.query('is_male == True and has_diabetes == True').value_counts().shape[0]
mujeres_diabe = dataframe.query('is_male == False and has_diabetes == True').value_counts().shape[0]
hombres_muertos = dataframe.query('is_male == True and is_dead == True').value_counts().shape[0]
mujeres_muertas = dataframe.query('is_male == False and is_dead == True').value_counts().shape[0]
hombres_fum = dataframe.query('is_male == True and is_smoker == True').value_counts().shape[0]
mujeres_fum = dataframe.query('is_male == False and is_smoker == True').value_counts().shape[0]

labels = ['Anemicos', 'Diabeticos' , 'Fumadores' , 'Muertos']
plt.bar(labels,[hombres_anemi,hombres_diabe,hombres_fum,hombres_muertos],width=-0.35,align='edge',color='b',label='Hombres')
plt.bar(labels,[mujeres_anemi,mujeres_diabe,mujeres_fum,mujeres_muertas],width=0.35,align='edge',color='r',label='Mujeres')

plt.title('Histograma agrupado por sexo')
plt.ylabel('Cantidad')
plt.xlabel('Categorias')
plt.legend(loc='upper right', fontsize='small')
plt.legend()
plt.show()

