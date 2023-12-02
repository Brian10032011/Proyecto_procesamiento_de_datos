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


'''
Parte 8 : Analizando distribuciones 2
'''
#Realizacion de grupos segun categorias
si_anemi = dataframe.query('has_anaemia').shape[0]
no_anemi = dataframe.query('has_anaemia==False').shape[0]
si_diabe = dataframe.query('has_diabetes').shape[0]
no_diabe = dataframe.query('has_diabetes == False').shape[0]
si_fum = dataframe.query('is_smoker').shape[0]
no_fum = dataframe.query('is_smoker == False').shape[0]
si_muertos = dataframe.query('is_dead == True').shape[0]
no_muertos = dataframe.query('is_dead == False').shape[0]

'''
CREACION DE SUBPLOTS
'''
#SUBPLOT DE ANEMICOS
plt.subplot(1,4,1)
categorias_anemi = ['No', 'Si']
valores_anemi = [no_anemi,si_anemi]

colors = ['lightcoral', 'lightgreen']
explode = [0.1, 0]
startangle = 90

plt.pie(valores_anemi, labels=categorias_anemi, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
plt.title('Anemicos')
plt.axis('equal') 

#SUBPLOT DE DIABETICOS
plt.subplot(1,4,2)
categorias_diabe = ['No', 'Si']
valores_diabe = [no_diabe,si_diabe]

colors = ['lightcoral', 'lightgreen']
explode = [0, 0]
startangle = 90

plt.pie(valores_diabe, labels=categorias_diabe, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
plt.title('Diabeticos')
plt.axis('equal') 

#SUBPLOT FUMADORES
plt.subplot(1,4,3)
categorias_fum = ['No', 'Si']
valores_fum = [no_fum,si_fum]

colors = ['lightcoral', 'lightgreen']
explode = [0.1, 0]
startangle = 90

plt.pie(valores_fum, labels=categorias_fum, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
plt.title('Fumadores')
plt.axis('equal') 

#SUBPLOT MUERTOS
plt.subplot(1,4,4)
categorias_muer = ['No', 'Si']
valores_muer = [no_muertos,si_muertos]

colors = ['lightcoral', 'lightgreen']
explode = [0.1, 0]
startangle = 90

plt.pie(valores_muer, labels=categorias_muer, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
plt.title('Muertos')
plt.axis('equal') 

plt.show()




