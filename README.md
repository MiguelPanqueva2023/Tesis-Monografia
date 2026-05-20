# Tesis-Monografia: MODELAMIENTO Y CONTROL DE UN CASO DE ESTUDIO SOBRE EL RIEGO DEL CULTIVO DE PAPA EN COLOMBIA

Este repositorio contiene el ecosistema de software, datos y documentación científica correspondiente al trabajo de grado titulado "Modelamiento y Control de un Caso de Estudio sobre el Riego del Cultivo de Papa en Colombia", desarrollado para optar al título de Ingeniero de Sistemas en la Universidad Distrital Francisco José de Caldas.

El proyecto implementa un enfoque de Caja Gris (Grey-Box Modeling) acoplado a un Control Predictivo Basado en Modelo No Lineal (NMPC) para optimizar la regulación hídrica en sistemas de cultivo de papa, mitigando desajustes paramétricos frente a variables agroclimáticas reales de la estación Tibaitatá (IDEAM).

# 📑 Estructura del Repositorio

El repositorio se encuentra organizado de manera modular para garantizar la reproducibilidad de los experimentos y el acceso directo a los componentes teóricos:

```Plaintext

├── 📁 code/
│   ├── 📁 linea_base_fija/          # Estrategia original con setpoint constante
│   │   ├── script1_dataset.m        # Generación de dataset base via NMPC nominal
│   │   ├── script2_train_ann.m      # Backpropagation manual para residuos fijos
│   │   └── script3_validation.m     # Lazo cerrado: Planta real + NMPC + ANN (Caja Gris)
│   │
│   └── 📁 linea_alternativa/        # Estrategia adaptativa con setpoint internacional variable
│       ├── script1_alt_dataset.m    # Generación de dataset con rampa dinámica (USA->ARG->COL)
│       ├── script2_alt_train_ann.m  # Entrenamiento matricial enfocado en múltiples regímenes
│       └── script3_alt_validation.m # Validación del lazo híbrido adaptativo transicional
│
├── 📁 data/
│   └── Datos_IDEAM_Tibaitata_2015_SEQUIA_AJUSTADA.xlsx  # Serie agroclimática histórica base
│
├── 📁 dependencies/                 # Toolkit y librerías externas requeridas
│   └── casadi-3.7.2-windows64-matlab2018b.zip  # Binarios oficiales de CasADi
│
├── 📁 references/                  # Biblioteca digital de soporte (Formatos PDF)
│   └── [Listado de archivos PDF indexados según la bibliografía oficial]
│
└── 📁 document/
    └── monografia_final.pdf         # Versión final del documento de la monografía de grado
```
⚠️ Nota sobre los archivos de datos (.mat): Los archivos binarios de datos resultantes (dataset_hibrido.mat, dataset_hibrido_refdin.mat y las matrices de pesos sinápticos entrenados red_entrenada.mat) no se incluyen en la carpeta /data para mantener el repositorio limpio, ya que se generan de forma automática y secuencial al ejecutar las simulaciones.

# 🚀 Instrucciones de Ejecución y Replicabilidad

Para recrear con éxito las simulaciones y el entrenamiento de las redes en su máquina local, siga estrictamente las siguientes directrices:

1. Directorio de Trabajo Único: Asegúrese de colocar el archivo de datos Excel (Datos_IDEAM_Tibaitata_2015_SEQUIA_AJUSTADA.xlsx) dentro del mismo directorio de trabajo de MATLAB desde el cual va a ejecutar los scripts de la línea correspondiente (o configure correctamente el Current Folder de MATLAB). Los scripts leen este archivo de forma local; si no se encuentra en la misma ruta de ejecución, el programa arrojará un error de lectura de archivo.

2. Configuración de Dependencias (CasADi): El repositorio incluye el toolkit de optimización necesario en la carpeta /dependencies. Para utilizarlo:

    Descomprima el archivo casadi-3.7.2-windows64-matlab2018b.zip en su máquina local.

    Al abrir el Script 1 o el Script 3 de cualquiera de las dos líneas, asegúrese de modificar la primera línea de código (addpath(genpath('...'))) colocando la ruta física absoluta de la carpeta que acaba de extraer para que MATLAB pueda importar CasADi e IPOPT correctamente.

3. Orden de Ejecución Obligatorio (Pipeline): Para cualquiera de las dos líneas de investigación, los scripts deben correrse en el siguiente orden jerárquico:

   - Script 1: Lee el Excel del IDEAM, calcula la ETo (FAO-56), simula el lazo predictivo nominal y exporta de forma automática el dataset de residuos físicos.

   - Script 2: Toma el dataset recién generado por el paso anterior, realiza el barajado estocástico, ejecuta el Backpropagation matricial manual y exporta el archivo con los pesos de la ANN optimizada.

   - Script 3: Cierra el lazo definitivo de control cargando los pesos de la red del paso anterior para ejecutar la validación robusta del controlador híbrido de Caja Gris.
  
# ⚙️ Resumen de Componentes del Software

Fase,Archivo,Descripción Matemática / Operacional
Fase 1,Script 1: Generación de Dataset,"Corre el lazo cerrado nominal contra la planta real para minar y consolidar los residuos dinámicos no modelados (Delta theta, Delta B) provocados por desajustes físicos del suelo y evapotranspiración (FAO-56)."
Fase 2,Script 2: Entrenamiento ANN,Inicializa y optimiza mediante Backpropagation matricial manual (sin toolboxes opacos) una red neuronal MISO (5 -> 20 -> 10 -> 2) con funciones de activación tanh para mapear el comportamiento del residuo físico.
Fase 3,Script 3: Validación Híbrida,"Cierra el lazo definitivo reinyectando la predicción en tiempo real de la ANN al NMPC, consolidando el modelo de Caja Gris que compensa las perturbaciones en el sistema físico de cultivo."


# 📚 Repositorio de Referencias e Investigación

En el directorio /references se encuentra consolidada la biblioteca científica en formato PDF recopilada y utilizada para la sustentación teórica, el modelado biofísico y la sintonización de los algoritmos de este proyecto. Entre los ejes temáticos clave incluidos destacan:

- Modelado de Cultivo e Hidrodinámica: Literatura estándar de la FAO (Boletín FAO-56 para Evapotranspiración), ecuaciones de transporte de humedad y curvas de retención hídrica en suelos.

- Algoritmia de Control Predictivo: Documentación sobre optimización no lineal no convexa e implementación de filtros de líneas de búsqueda mediante IPOPT (Wächter & Biegler).

- Machine Learning de Caja Gris: Artículos científicos enfocados en el acoplamiento de redes neuronales artificiales a leyes físicas mecánicas preexistentes para la predicción del estado del suelo.

# 🎓 Autores y Dirección

  - Andrés Felipe Arias Guevara – Código: 20192020147
  - Miguel Angel Panqueva Pulido – Código: 20201020174

  Director del Proyecto:
  - Duván Andrés Téllez Castro
  
  Facultad de Ingeniería, Ingeniería de Sistemas, Universidad Distrital Francisco José de Caldas.

