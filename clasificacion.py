import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Cargando Dataset
df = pd.read_csv("resultado_df_limpio.csv")
dataframe = pd.DataFrame(df)

'''
Parte 11 : Clasifiacion Parte 1
'''
#Clasificacion personas muertas y no muertas
personas_muertas = dataframe.query('is_dead == True').value_counts().shape[0]
personas_no_muertas = dataframe.query('is_dead == False').value_counts().shape[0]

#Graficando distribucion de clases
labels = ['']
plt.bar(labels,[personas_no_muertas],width=-0.35,align='edge',color='b',label='0')
plt.bar(labels,[personas_muertas],width=0.35,align='edge',color='r',label='1')

plt.title('Distribucion de Clases')
plt.ylabel('Cantidad de personas')
plt.xlabel('Estado de Vida (0: No muerto, 1: Muerto)')
plt.legend(loc='upper right', fontsize='small')
plt.legend()
plt.show()

#Eliminando columna CATEGORIA EDAD
df = dataframe.drop('Categoría de Edad',axis = 1)
print(df)

# Is dead es la columna de clases en mi Dataframe
X = df.drop('is_dead', axis=1)
y = df['is_dead']

# Realizando la partición estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Imprimir las formas de los conjuntos de entrenamiento y test
print("Forma de X_train:", X_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de y_test:", y_test.shape)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1)
tree.fit(X_train, y_train)

y_pred_train = tree.predict(X_train)
y_pred_test = tree.predict(X_test)

print("Precisión del modelo en train:", accuracy_score(y_train, y_pred_train))
print("Precisión del modelo en test:", accuracy_score(y_test, y_pred_test))
print("Profundidad:", tree.get_depth())
print("Hojas:", tree.get_n_leaves())