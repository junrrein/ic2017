# Introducción y Perceptrón Simple

* ¿Qué problemas resultan fáciles de resolver por una persona? ¿Cuáles por una computadora?
* ¿Qué capacidades de las personas nos gustaría replicar de forma computacional?
* Historia de la inteligencia artificial:
    * ¿Qué modelamos?
    * ¿Cómo lo modelamos?
    * Inteligencia Computacional - Métodos destacados.
* Perceptrón simple
    * Inspiración biológica
        * Características de las redes neuronales naturales
            * No linealidad
            * Paralelismo
            * Aprendizaje, adaptabilidad
            * Generalización
            * Robustez
        * La neurona biológica: Soma, dendritas, axón
        * Fisiología de la neurona
            * Sinapsis
            * Despolarización, comportamiento todo/nada
            * Propagación del impulso
            * Refuerzo de las sinapsis / aprendizaje
        * Modelo simplificado de neurona: Enlaces conectores, sumador, función de activación, umbral
            * Trazar el paralelo de cada componente a lo biológico
    * Perceptrón simple
        * Modelo matemático: Suposiciones, fórmula
        * Funciones de activación (signo, lineal, sigmoidea, otras)
        * Ejemplo 2D de cómo un perceptrón divide el espacio de entradas e hiperplano de decisión
        * Ejemplo del OR - Necesidad del sesgo
            * Cálculo directo de los pesos
        * Entrenamiento por corrección de error
            * Fórmula de actualización de pesos
            * Principio de mínima perturbación
            * Tasa de aprendizaje
        * Entrenamiento por método del gradiente
            * Fórmula para activación lineal
            * Fórmula para activación sigmoidea
* Ejemplos de arquitecturas neuronales
    * Redes feed-forward
    * Redes recurrentes
    * Modelos híbridos
* Procesos de aprendizaje (estudiar del resumen)
    * Tipos
    * Reglas

# Capacidad de generalización

* Superficies de error
    * Qué son: La forma de la función de error para una red neuronal cuando dicha función es evaluada para distintos valores de los parámetros de la red (pesos sinápticos). $ \varepsilon = f(w_1, w_2, \dots, w_n) $
    * Ejemplos
    * ¿Cómo nos movemos en la superficie de error?
    * ¿Es siempre fácil encontrar el mínimo global usando un perceptrón? ¿Es deseable encontrarlo? (Relación con la capacidad de generalización)
* Sobre-entrenamiento. ¿Por qué surge? ¿Cómo lo evitamos? Criterio de detención temprana y particionamiento entrenamiento - monitoreo - prueba. ¿Por qué no nos quedamos solo con entrenamiento - prueba?
* Validación cruzada. Particionamiento. Variantes
    * Aleatorizar el orden de los patrones. ¿Para qué se hace esto?
    * ¿Por qué la validación cruzada nos permite evaluar la capacidad de generalización de una red? ¿Por qué el hacer varias particiones ayuda?
    * Método de detención temprana de entrenamiento
    * Leave k-out y Leave one-out: Diferencias en el promedio y el desvío del error medido
* Medidas desempeño
    * Matriz de confusión
    * Sensibilidad: Si entra un positivo, ¿lo reconoce como tal? Ejemplo: Si tengo un sistema para detectar casos de cáncer, si hay alguien que tiene, me interesa mucho que me lo detecte
    * Especificidad: Si entra un negativo, ¿lo reconoce como tal? Ejemplo: Un sistema anti-spam.
    * Precisión: Qué tasa de los identificados como positivos de verdad son positivos. En el ejemplo del detector de cáncer, ¿cuántos de los que digo que tienen cáncer de verdad lo tienen?
    * Exactitud: Tasa global de aciertos
* Error relativo y medidas de desempeño al comparar dos clasificadores
* Errores de predicción en series. Distintas medidas de error

# Perceptrón multicapa

* ¿Qué significa que un problema sea linealmente separable?
* Problema XOR con 3 neuronas. Primera capa. Segunda capa. Tablas de verdad para cada neurona
* Regiones de decisión
* Arquitectura. Capas, pesos y salidas de cada capa
* Cálculo de las salidas de cada capa
* Propagación hacia atrás
    * Criterio de error
    * Aplicación del gradiente - Caso general (fórmulas)
    * Gradiente del error local instantáneo ($ \delta $)
    * Derivada de la función sigmoidea
    * $ \delta $ de la capa de salida. Actualización de pesos
    * $ \delta $ de las capas ocultas. Actualización de pesos. Fórmula general para la capa $ p $
    * Resumen del algoritmo
    * Término de momento. Efecto en direcciones largas de descenso y en mínimos locales (en la función de error)
* ¿Para qué problemas hacen falta más capas?
* ¿Cómo entrenar un MLP por lotes?

# Redes con RBFs

* Funciones radiales vs. sigmoideas. Regiones radiales
* Origen de las RBFs como aproximador de funciones (de salida real). Modelo matemático
* Arquitectura de una RBF-NN. Modelo matemático
* Métodos de entrenamiento
    * Método 1: Entrenamiento mixto
        * Etapa no supervisada. Utilizando k-medias, SOM y otros.
        * Etapa supervisada
    * Método 2: Entrenamiento supervisado
* Método 1 - k-medias
    * Generalidades
    * k-medias por lotes
    * k-medias online
* Método 1 - Segunda parte del entrenamiento - Parte supervisada. Métodos de entrenamiento. ¿Por qué no se usa usualmente el entrenamiento por pseudo-inversa?
    * Derivación de la fórmula de actualización de pesos
* Comparación entre RBF-NNs y MLPs
* Gaussianas n-dimensionales. Casos simplificados. Caso general

# Mapas auto-organizativos

* Auto-organización. Ejemplos
* Arquitectura de un SOM. Salida
* Vecindades
    * Formas básicas
    * Alcance (radio de vecindad)
    * Excitaciones asociadas al entorno
    * Ejemplos de funciones de vecindad
* Entrenamiento
    * Características principales
        * Entrenamiento no supervisado
        * Aprendizaje competitivo
    * Algoritmo de entrenamiento
        * Consideraciones prácticas sobre forma de vencidad, tamaño de la misma y velocidad de aprendizaje
        * Etapas del entrenamiento
            * Ordenamiento topológico: El resultado de esta etapa es que neuronas cercanas en el mapa bidimensional de neuronas mapean a conjuntos de patrones cercanos en el espacio n-dimensional de entrada.
* Como utilizar un SOM para clasificar patrones
    * Entrenamiento no supervisado
    * Etitquetado de neuronas
    * Clasificación por mínima distancia

# Cuantización vectorial con aprendizaje (LVQ)

* Conceptos básicos
    * Cuantizador escalar
    * Cuantizador vectorial
    * Cómo entrenarlo
* Algoritmo LVQ1
    * Observaciones
        * Interpretación gráfica
        * Arquitectura
        * Comparación con SOM
    * Velocidad de aprendizaje
    * LVQ1-O

# Redes dinámicas

* Introducción
    * ¿Por qué dinámicas?
    * Aproximaciones
    * Caso general
* Clasificación
    * Redes con retardos en el tiempo
    * Redes recurrentes
        * Totalmente recurrentes
        * Parcialmente recurrentes
* Redes de Hopfield
    * Arquitectura
    * Modelo matemático
    * Generalidades
    * Entrenamiento (aprendizaje Hebbiano). Observaciones
    * Prueba (recuperación). Observaciones
    * Campos energéticos de Hopfield
* Retropropagación a través del tiempo
    * Arquitectura
    * Expansión en forma de red feed-forward
    * Consideraciones del entrenamiento
        * Cantidad de pesos que se entrenan
        * Cantidad de pasos hacia atrás que tomar (truncado)
* Redes neuronales con retardos en el tiempo (TDNN)
    * Arquitectura
    * Clasificación espacio-temporal
* Redes de Elman y Jordan. Arquitectura
    * Motivación de redes de Elman:

        La mayoría de las arquitecturas propuestas por Elman eran recurrentes y diseñadas para aprender patrones secuenciales o variantes en el tiempo. El principal interés de Elman en esta arquitectura fue para algoritmos de procesamiento del lenguaje, pero se dió cuenta que podía servir para cualquier caso que involucre secuencias.

        Para que la red aprenda una secuencia, Elman definió un contexto compuesto por estados internos anteriores, y entonces añadió una capa de "unidades de contexto" a una red feed-forward estándar. Estas unidades de contexto proveen a la capa oculta de una historia de sus activaciones previas, y le permite a la red aprender tareas que se extienden en el tiempo.

# Lógica

* ¿Qué es una lógica?
* Tipos de lógica. Elementos del mundo. Estados que asume el conocimiento
* Reglas de inferencia
* Componentes de un sistema experto
    * Memoria de trabajo
    * Memoria de producciones
* Fases de un sistema experto
    * Cotejo
        * Conjunto de conflicto
    * Resolución de conflictos
        * Criterios de selección de reglas
    * Aplicación de regla
* Análisis de ejemplo: zoológico

# Lógica borrosa

* introducción
    * Reglas lingüísticas
    * Incerteza vs. aleatoriedad
* Conjuntos binarios tradicionales
* Conjuntos borrosos
    * Conjuntos binarios (como conjuntos borrosos especiales)
* Operaciones con conjuntos borrosos
    * Subconjunto borroso
    * Inclusión
    * Igualdad
    * Complemento
    * Intersección
    * Unión
    * Suma disyuntiva
    * Diferencia
    * Distancia
        * Hamming (y Hamming relativa)
        * Euclídea (y euclídea relativa)
* Caracterización de conjuntos borrosos
    * Conjunto binario de nivel $ \alpha $
    * Conjunto convexo
    * Conjunto normal
    * Tamaño de un conjunto borroso
    * Propiedades de las operaciones entre un conjunto y su complemento
    * Conjunto borroso medio
        * Propiedad con su complemento
* Entropía borrosa
    * Definición
    * Teorema de la entropía borrosa
    * Teorema del subconjunto borroso
    * Relación entre los teoremas

# Memorias asociativas borrosas

* ¿Cómo construimos un sistema borroso? Reglas. Diagrama en bloques. Diagrama de conjuntos
* Composición de reglas. Modelo aditivo estándar
* Conjuntos borrosos de entrada. Fuzzyficacion de la entrada
* Codificación de reglas borrosas
    * Correlación mínimo
    * Correlación producto
    * Consideraciones prácticas. Conjuntos binarios de entrada
* Composición de reglas
    * Superposición de reglas codificadas en matrices. Problema que tiene este método
    * Modelo aditivo estándar
    * Modelo aditivo estándar con factor de activación. Factores mínimo y escala
    * Combinación de conjuntos. Formas de hacerlo
* Defuzzyficacion. Tipos (máximo y centroide)
* Conjuntos de pertenencia continuos
    * Fornma y codificación de conjuntos borrosos
    * Composición y defuzzyficacion en un solo paso, usando centroides y áreas
* Representación de dos antecedentes por consecuente
* Composición de dos antecedentes de una regla
* Comparación con con redes neuronales y sistemas expertos
    * Entradas incompletas o corruptas
    * Visibilidad del model
    * Estructuración del conocimiento almacenado
    * Forma de aprendizaje
    * Tamaño de la representación

# Búsqueda

* Deficiones: Estados. Formulación de objetivos. Formulación del problema. Acciones. Solución. Búsqueda
* Problemas bien definidos
    * Estado inicial
    * Conjunto de acciones/operadores disponibles
    * Espacio de estados del problema
    * Test de verificación del objetivo
    * Función de evaluación de costo del camino
* Definición de proceso de búsqueda
* Nodo de un árbol de búsqueda. Posible estructura
* Algoritmo general de búsqueda
* Estrategias de búsqueda
    * Criterios de evaluación de una estrategia (son 4)
    * Diferencia entre búsqueda informada y no informada
    * Estrategias no informadas (mencionamos 6, definimos 3)
    * Estrategias informadas (son 2)
        * Función de evaluación. ¿Qué restricción debe cumplir
        * ¿Bajo qué condiciones se puede demostrar que la estrategia A* es óptima y completa? Ejemplos

# Computación evolutiva - Algoritmos genéticos

* Ideas de Lamarck y Darwin
* Ideas generales
    * Poblaciones vs. individuos
    * Mejores vs. "adaptados"
    * Aleatoriedad en la selección natural
    * Diversidad y operadores de variación en la población
* La evolución como un algoritmo. Elementos de un algoritmo evolutivo
* Algoritmos genéticos
    * Representación de los individuos. Genotipo y fenotipo. Ejemplos
    * Función de aptitud
        * ¿Qué hace? ¿Para qué sirve?
        * Características generales
            * Monotonicidad
            * Precisión
            * Suavidad regulable
            * Penalización de complejidad
        * Ejemplos: Promedios de error. Estadísticas. Medidas de información. Otras. Ejemplos prácticos (para ubicar figuras en un área; para entrenar una red neuronal, etc.)
    * Operadores
        * Selección
            * Características que deben poseer
            * Ejemplos
                * Rueda de ruleta. Problemas que tiene (y posibles soluciones). 
                * Ventanas
                * Competencia
        * Variación. Mutación y cruza. ¿Qué funciones cumplen en la búsqueda de solución?
        * Reemplazo durante la reproducción
            * Total
            * Brecha generacional
            * Elitismo. Consecuencias en la función de aptitud
    * Características principales
        * Búsqueda en un espacio codificado de parámetros. ¿Meter acá lo de convergencia asegurada (teorema de esquemas)?
        * Búsqueda en múltiples puntos del espacio de soluciones
        * Pocos requisitos sobre la función objetivo
        * Algoritmo de naturaleza estocástica
        * La estructura de los operadores los hace muy efectivos al realizar búsquedas globales
        * Múltiples objetivos
        * Desventajas. ¿Dónde está el cuello de botella?
* Métodos tradicionales vs. algoritmos evolutivos
    * Parámetros a determinar. ¿Cómo los trabaja?
    * Infomarción necesaria de la función objetivo
    * Reglas de transición
    * Exploración del espacio de soluciones
* Esquemas de paralelismo
    * Maestro-esclavos
    * Colonias
    * Celdas
* Parámetros que controlan la evolución (son 5).
* Estrategias de evolución
    * Representación fenotípica. Variables objetivo y de control
    * Función de aptitud
    * Operadores: Mutación, cruza y selección.
    * Reproducción (dos mecanismos)
* Representación de los individuos
    * Genético
        * Muchos genes con pocos alelos
        * Epistasis
        * Representación lejana al dominio del problema
        * Gran cantidad de soluciones inválidas en la población
    * Evolutivo
        * Pocos genes con muchos alelos
        * Convergencia muy dependiente de los operadores
        * Necesidad de redefinición de los operadores
    * Otras representaciones
* Programación genética
    * Elementos básicos de un programa
    * Representación en árbol de un programa
    * Cruzas: En base al intercambio de ramas
    * Mutaciones: En base al reemplazo con un árbol generado al azar
* Restricciones del problema
    * Redefinición de la representación de forma de que siempre se generen fenotipos válidos
    * Rechazo o eliminación de individuos inválidos. ¿Cuándo sirve?
    * Reparación del material genético
    * Modificación de los operadores de variación
    * Esquemas de penalización en la función de aptitud

# Inteligencia colectiva

* Autómatas de estados finitos
    * Definición: Entradas, salidas, estados y reglas de transición
    * Forma de determinar la salida: Función de salida y otras
    * Representación gráfica de estados y reglas de transición
    * Reglas de transición determinísticas y aleatorias
* Autómatas celulares
    * Definición: Autómatas, topología y conectividad
    * Topología: tipos
    * Conectividad
        * Tipos y tamaños de vecindad
        * Tipos de conexiones
    * Identificar todos estos elementos en un ejemplo
* Agentes
    * Agente
        * Definición
        * Sensores y efectores
        * Autonomía (capacidad de aprender de su experiencia y de modificar su comportamiento)
    * Agente inteligente: Debe ser proactivo, reactivo, y poseer habilidad social
    * Agente racional: Realiza acciones correctas
        * Agente racional ideal: Capaz de percibir, conocer, decidir y actuar
    * Sistemas multi-agente
        * Son cooperativos por varios mecanismos (son 3)
        * ¿Cuándo un agente posee racionalidad social? Utilidad esperada
* Inteligencia colectiva
    * Características generales
        * Auto-organización
        * Estigmergía
        * Comportamiento emergente: inteligencia distribuida, robustez
        * Fuerte interacción local
        * Organización social altamente estructurada
        * Colaboración versus competencia
        * Componentes estocásticas
        * Bio-inspiración
    * Ejemplos en la naturaleza
    * Elementos individuales: Boids, autómatas, agentes. Neuronas también pueden verse acá.
    * Ejemplos de algoritmos en inteligencia evolutiva
    * Aplicaciones: Optimización, entrenamiento, identificación, búsqueda, agrupamiento, clasificación

# Colonias de hormigas

* La inspiración biológica
    * ¿Cómo encuentran las hormigas una fuente de comida? Papel de las feromonas
    * Comunicación indirecta, estigmergía
    * Ilustrar lo anterior con el experimento del puente binario
* Algoritmos
    * Definiciones: $ G = (V, E) $;  $ \sigma_{ij} $;  $ k = 1, 2, \dots, N $;  $ \mathcal{N}_i $;  $ \mathbb{p}^k(t) $
    * Algoritmo: Colonia de hormigas simple (sACO)
    * Algoritmo: Sistema de hormigas (AS)
    * Diferencias entre los dos anteriores
        * Cálculo de la probabilidad de selección del próximo nodo
        * Forma de tratar bucles en el camino
        * Cantidad de feromonas a depositar

# Enjambre de partículas

* La inspiración biológica
    * Ejemplo en la naturaleza de la bandada de pájaros
* Definiciones
    * Posición
    * Velocidad
    * Regla básica de actualización de posición
    * Mejor posición personal
    * Función de error o aptitud
    * Mejor global: Topología, mejor posición global
    * Mejor local: Topología, selección del entorno, mejor posición en el entorno local
    * Función de la componente cognitiva y la componente social en la exploración del espacio de soluciones. Rol del $ \mathbb{r} $. Relación con cruzas y mutaciones de algoritmos genéticos
* Algoritmo: Enjambre del mejor global (gEP)
* Algoritmo: Enjambre del mejor local (lEP)

