# Empezando

Dos preguntas clave para quienes exploran por primera vez la GP son:
1. ¿Qué debo leer para iniciarme en GP?

2. ¿Debo implementar mi propio sistema de GP o debo usar un paquete existen
te? Si es así, ¿qué paquete debo usar?

# Programación Genética en Pocas Palabras

- Crossover: La creación de un programa hijo combinando partes elegidas 
al azar de dos programas padres seleccionados.
- Mutation: La creación de un nuevo programa hijo alterando aleatoriamente una parte elegida al azar de un programa padre seleccionado

# Prerequisites

1: Cree aleatoriamente una población inicial de programas a partir de las primitivas disponibles (más sobre esto en la Sección 2.2). 

2: repetir 

3: Ejecute cada programa y determine su idoneidad. 

4: Seleccione uno o dos programas de la población con una probabilidad basada en la aptitud para participar en operaciones genéticas (Sección 2.3). 

5: Cree nuevos programas individuales aplicando operaciones genéticas con probabilidades especificadas (Sección 2.4). 

6: hasta que se encuentre una solución aceptable o se cumpla alguna otra condición de parada (por ejemplo, se alcanza un número máximo de generaciones). 

7: devuelva el mejor individuo hasta ahora

# Aridad

En algunos casos, puede ser deseable utilizar primitivas GP que acepten un número variable de argumentos (una cantidad que llamaremos arity)

# Preparándose para ejecutar programación genética

Para aplicar un sistema de GP a un problema, es necesario tomar varias decisiones; estos a menudo se denominan pasos preparatorios. Las opciones clave son: 
1. ¿Cuál es el conjunto de terminales? 
2. ¿Cuál es el conjunto de funciones? 
3. ¿Cuál es la medida de aptitud? 
4. ¿Qué parámetros se utilizarán para controlar la ejecución? 
5. ¿Cuál será el criterio de terminación y qué se considerará como resultado de la ejecución?

<div >
    <img src="/img/1.png">
</div>


