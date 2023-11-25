# Proyecto_procesamiento_de_datos
El proyecto de este curso consiste en analizar el conjunto de datos introducido en esta sección, procesarlo, limpiarlo y finalmente ajustar modelos de machine learning para realizar predicciones sobre estos datos.

Para el desarrollo de esta etapa del proyecto necesitamos intalar la librería datasets de Huggingface

pip install datasets
Vamos a trabajar con un dataset sobre fallo cardíaco

El dataset contiene registros médicos de 299 pacientes que padecieron insuficiencia cardíaca durante un período de seguimiento.

Las 13 características clínicas incluidas en el conjunto de datos son:

Edad: edad del paciente (años)
Anemia: disminución de glóbulos rojos o hemoglobina (booleano)
Presión arterial alta: si el paciente tiene hipertensión (booleano)
Creatinina fosfoquinasa (CPK): nivel de la enzima CPK en la sangre (mcg/L)
Diabetes: si el paciente tiene diabetes (booleano)
Fracción de eyección: porcentaje de sangre que sale del corazón en cada contracción (porcentaje)
Plaquetas: plaquetas en la sangre (kiloplaquetas/mL)
Sexo: mujer u hombre (binario)
Creatinina sérica: nivel de creatinina sérica en la sangre (mg/dL)
Sodio sérico: nivel de sodio sérico en la sangre (mEq/L)
Fumar: si el paciente fuma o no (booleano)
Tiempo: período de seguimiento (días)
[Objetivo] Evento de fallecimiento: si el paciente falleció durante el período de seguimiento (booleano)


Proyecto integrador procesamiento de datos
