import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
'''
LEER Y CONVERTIR A DATAFRAME EL ARCHIVO LIMPIO
'''
df = pd.read_csv("resultado_df_limpio.csv")
dataframe = pd.DataFrame(df)
print(dataframe)

# '''
# 1.GRAFICANDO LA DISTRIBUCION DE LAS EDADES EN UN HISTOGRAMA  
# '''
# edades = dataframe['age'].unique()
# len(edades)

# plt.hist(edades, bins = 10 , edgecolor = 'black')
# plt.title('Distribución de Edades')
# plt.xlabel('Edades')
# plt.ylabel('Cantidad de personas')
# plt.show()

# '''
# 2. GRAFICA DE HISTOGRAMA AGRUPADO POR HOMBRE Y MUJER
# '''
# hombres_anemi = dataframe.query('is_male == True and has_anaemia == True').value_counts().shape[0]
# mujeres_anemi = dataframe.query('is_male == False and has_anaemia == True').value_counts().shape[0]
# hombres_diabe = dataframe.query('is_male == True and has_diabetes == True').value_counts().shape[0]
# mujeres_diabe = dataframe.query('is_male == False and has_diabetes == True').value_counts().shape[0]
# hombres_muertos = dataframe.query('is_male == True and is_dead == True').value_counts().shape[0]
# mujeres_muertas = dataframe.query('is_male == False and is_dead == True').value_counts().shape[0]
# hombres_fum = dataframe.query('is_male == True and is_smoker == True').value_counts().shape[0]
# mujeres_fum = dataframe.query('is_male == False and is_smoker == True').value_counts().shape[0]

# labels = ['Anemicos', 'Diabeticos' , 'Fumadores' , 'Muertos']
# plt.bar(labels,[hombres_anemi,hombres_diabe,hombres_fum,hombres_muertos],width=-0.35,align='edge',color='b',label='Hombres')
# plt.bar(labels,[mujeres_anemi,mujeres_diabe,mujeres_fum,mujeres_muertas],width=0.35,align='edge',color='r',label='Mujeres')

# plt.title('Histograma agrupado por sexo')
# plt.ylabel('Cantidad')
# plt.xlabel('Categorias')
# plt.legend(loc='upper right', fontsize='small')
# plt.legend()
# plt.show()


# '''
# Parte 8 : Analizando distribuciones 2
# '''
# #Realizacion de grupos segun categorias
# si_anemi = dataframe.query('has_anaemia').shape[0]
# no_anemi = dataframe.query('has_anaemia==False').shape[0]
# si_diabe = dataframe.query('has_diabetes').shape[0]
# no_diabe = dataframe.query('has_diabetes == False').shape[0]
# si_fum = dataframe.query('is_smoker').shape[0]
# no_fum = dataframe.query('is_smoker == False').shape[0]
# si_muertos = dataframe.query('is_dead == True').shape[0]
# no_muertos = dataframe.query('is_dead == False').shape[0]

# '''
# CREACION DE SUBPLOTS
# '''
# #SUBPLOT DE ANEMICOS
# plt.subplot(1,4,1)
# categorias_anemi = ['No', 'Si']
# valores_anemi = [no_anemi,si_anemi]

# colors = ['lightcoral', 'lightgreen']
# explode = [0.1, 0]
# startangle = 90

# plt.pie(valores_anemi, labels=categorias_anemi, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
# plt.title('Anemicos')
# plt.axis('equal') 

# #SUBPLOT DE DIABETICOS
# plt.subplot(1,4,2)
# categorias_diabe = ['No', 'Si']
# valores_diabe = [no_diabe,si_diabe]

# colors = ['lightcoral', 'lightgreen']
# explode = [0, 0]
# startangle = 90

# plt.pie(valores_diabe, labels=categorias_diabe, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
# plt.title('Diabeticos')
# plt.axis('equal') 

# #SUBPLOT FUMADORES
# plt.subplot(1,4,3)
# categorias_fum = ['No', 'Si']
# valores_fum = [no_fum,si_fum]

# colors = ['lightcoral', 'lightgreen']
# explode = [0.1, 0]
# startangle = 90

# plt.pie(valores_fum, labels=categorias_fum, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
# plt.title('Fumadores')
# plt.axis('equal') 

# #SUBPLOT MUERTOS
# plt.subplot(1,4,4)
# categorias_muer = ['No', 'Si']
# valores_muer = [no_muertos,si_muertos]

# colors = ['lightcoral', 'lightgreen']
# explode = [0.1, 0]
# startangle = 90

# plt.pie(valores_muer, labels=categorias_muer, colors=colors, explode=explode, startangle=startangle, shadow=True, autopct='%1.1f%%')
# plt.title('Muertos')
# plt.axis('equal') 

# plt.show()


# '''
# Parte 9 : Analizando distribuciones parte 3
# '''
# #Eliminando columnas
# eliminar_is_dead = dataframe.drop('is_dead',axis = 1)
# eliminar_categoria = eliminar_is_dead.drop('Categoría de Edad',axis = 1)

# #Definiendo eje X
# X = eliminar_categoria.values

# #Convirtiendo a un array de NumPy y definiendo eje y
# dead_event = dataframe['is_dead'].values
# y = dead_event.reshape(-1,1)

# #Creando array de (224,3)
# X_embedded = TSNE(
#     n_components=3,
#     learning_rate='auto',
#     init='random',
#     perplexity=3
# ).fit_transform(X)

# #Creacion dataFrame con los resultado de TSNE
# df_tsne = pd.DataFrame(data=X_embedded, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
# df_tsne ['is_dead'] = y

# #Visualizacion en 3D
# fig = px.scatter_3d(df_tsne, x='Dimension 1', y='Dimension 2', z='Dimension 3',
#                     color='is_dead', title='Visualización grafica de dispersion vivos y muertos',
#                     labels={'is_dead': 'Estado (0: Vivo, 1: Muerto)'})


# fig.update_layout(scene=dict(zaxis=dict(range=[df_tsne['Dimension 3'].min(), df_tsne['Dimension 3'].max()])))
# fig.show()


'''
Parte 10 : Prediciendo datos de una columna
'''
#Eliminando columnas 'is_dead','age',''categoria de edad' para que sea la matriz X 
X = dataframe.drop(['is_dead','age','Categoría de Edad'],axis=1)

#Definiendo age como eje y ,  ajustando regresion lineal
y = dataframe['age']
regression = LinearRegression()
regression.fit(X, y)

#Comparar las edades reales con las predicciones e imprimirlas
edad_pred = regression.predict(X)
comparasion_edades = pd.DataFrame({'Edad Real': y, 'Edad Predicha': edad_pred})
print (comparasion_edades)

#Calculando el error cuadratico e imprimirlo
mse = mean_squared_error(y, edad_pred)
print ('Calculo error cuadratico :',mse)