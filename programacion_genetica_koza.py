"""
Programación Genética (PG) - Implementación basada en John Koza
==============================================================

Este ejemplo implementa un sistema de Programación Genética que evoluciona
expresiones matemáticas representadas como árboles para aproximar una función objetivo.

Inspirado en el trabajo pionero de John Koza en "Genetic Programming: On the Programming 
of Computers by Means of Natural Selection" (1992).

Autor: Ejemplo educativo
Fecha: 22 de octubre de 2025
"""

import random
import math
import copy
from typing import List, Union, Callable, Optional

class Nodo:
    """
    Representa un nodo en el árbol de expresión de PG.
    Puede ser una función (operador) o un terminal (variable/constante).
    """
    def __init__(self, valor: Union[str, float], hijos: Optional[List['Nodo']] = None):
        self.valor = valor
        self.hijos = hijos if hijos is not None else []
        self.es_terminal = len(self.hijos) == 0
    
    def evaluar(self, x: float) -> float:
        """
        Evalúa el nodo recursivamente con el valor de x dado.
        """
        if self.es_terminal:
            if self.valor == 'x':
                return x
            else:
                return float(self.valor)  # Constante numérica
        
        # Es una función, evaluar hijos primero
        valores_hijos = [hijo.evaluar(x) for hijo in self.hijos]
        
        # Aplicar la función correspondiente
        try:
            if self.valor == '+':
                return valores_hijos[0] + valores_hijos[1]
            elif self.valor == '-':
                return valores_hijos[0] - valores_hijos[1]
            elif self.valor == '*':
                return valores_hijos[0] * valores_hijos[1]
            elif self.valor == '/':
                # Protección contra división por cero
                if abs(valores_hijos[1]) < 1e-6:
                    return 1.0
                return valores_hijos[0] / valores_hijos[1]
            elif self.valor == 'sin':
                return math.sin(valores_hijos[0])
            elif self.valor == 'cos':
                return math.cos(valores_hijos[0])
            elif self.valor == 'exp':
                # Protección contra overflow
                try:
                    resultado = math.exp(min(valores_hijos[0], 100))
                    return resultado if not math.isnan(resultado) else 1.0
                except OverflowError:
                    return 1.0
            else:
                return 1.0  # Función desconocida
        except:
            return 1.0  # Error en evaluación, retornar valor seguro
    
    def copiar(self) -> 'Nodo':
        """
        Crea una copia profunda del nodo y sus descendientes.
        """
        hijos_copiados = [hijo.copiar() for hijo in self.hijos]
        return Nodo(self.valor, hijos_copiados)
    
    def __str__(self) -> str:
        """
        Representación en cadena del árbol (notación de prefijo).
        """
        if self.es_terminal:
            return str(self.valor)
        
        hijos_str = ' '.join([str(hijo) for hijo in self.hijos])
        return f"({self.valor} {hijos_str})"

class ProgramacionGenetica:
    """
    Implementación principal del algoritmo de Programación Genética.
    """
    
    def __init__(self, 
                 tamaño_poblacion: int = 100,
                 max_generaciones: int = 50,
                 profundidad_maxima: int = 6,
                 prob_cruce: float = 0.9,
                 prob_mutacion: float = 0.1,
                 tamaño_torneo: int = 3):
        """
        Inicializa los parámetros del algoritmo de PG.
        
        Args:
            tamaño_poblacion: Número de individuos en la población
            max_generaciones: Número máximo de generaciones
            profundidad_maxima: Profundidad máxima de los árboles
            prob_cruce: Probabilidad de cruce
            prob_mutacion: Probabilidad de mutación
            tamaño_torneo: Tamaño del torneo para selección
        """
        self.tamaño_poblacion = tamaño_poblacion
        self.max_generaciones = max_generaciones
        self.profundidad_maxima = profundidad_maxima
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.tamaño_torneo = tamaño_torneo
        
        # Conjunto de funciones (nodos internos)
        self.funciones = ['+', '-', '*', '/', 'sin', 'cos']
        self.aridad = {'+': 2, '-': 2, '*': 2, '/': 2, 'sin': 1, 'cos': 1, 'exp': 1}
        
        # Conjunto de terminales (nodos hoja)
        self.terminales = ['x'] + [round(random.uniform(-5, 5), 2) for _ in range(10)]
        
        # Datos de entrenamiento (función objetivo: x^2 + 2*x + 1)
        self.datos_x = [i * 0.1 for i in range(-50, 51)]  # x de -5 a 5
        self.datos_y = [x**2 + 2*x + 1 for x in self.datos_x]  # y = x^2 + 2x + 1
        
        self.poblacion = []
        self.mejor_individuo: Optional[Nodo] = None
        self.mejor_fitness = float('inf')
        self.historial_fitness = []
    
    def generar_arbol_aleatorio(self, profundidad_max: int, metodo: str = 'grow') -> Nodo:
        """
        Genera un árbol aleatorio usando el método especificado.
        
        Args:
            profundidad_max: Profundidad máxima del árbol
            metodo: 'grow' o 'full' (método de Koza)
        
        Returns:
            Nodo raíz del árbol generado
        """
        if profundidad_max <= 0 or (metodo == 'grow' and random.random() < 0.3):
            # Crear terminal
            terminal = random.choice(self.terminales)
            return Nodo(terminal)
        
        # Crear función
        funcion = random.choice(self.funciones)
        aridad_funcion = self.aridad[funcion]
        
        hijos = []
        for _ in range(aridad_funcion):
            hijo = self.generar_arbol_aleatorio(profundidad_max - 1, metodo)
            hijos.append(hijo)
        
        return Nodo(funcion, hijos)
    
    def inicializar_poblacion(self):
        """
        Inicializa la población usando el método 'ramped half-and-half' de Koza.
        Combina métodos 'grow' y 'full' para crear diversidad inicial.
        """
        self.poblacion = []
        
        # Ramped half-and-half: mitad con 'grow', mitad con 'full'
        for i in range(self.tamaño_poblacion):
            profundidad = random.randint(2, self.profundidad_maxima)
            metodo = 'grow' if i % 2 == 0 else 'full'
            individuo = self.generar_arbol_aleatorio(profundidad, metodo)
            self.poblacion.append(individuo)
    
    def calcular_fitness(self, individuo: Nodo) -> float:
        """
        Calcula el fitness de un individuo basado en el error cuadrático medio.
        Menor error = mejor fitness.
        
        Args:
            individuo: Árbol a evaluar
            
        Returns:
            Valor de fitness (error)
        """
        error_total = 0.0
        
        for x_val, y_esperado in zip(self.datos_x, self.datos_y):
            try:
                y_predicho = individuo.evaluar(x_val)
                if math.isnan(y_predicho) or math.isinf(y_predicho):
                    y_predicho = 1000.0  # Penalizar valores inválidos
                
                error = (y_esperado - y_predicho) ** 2
                error_total += error
            except:
                error_total += 1000.0  # Penalizar errores de evaluación
        
        # Error cuadrático medio
        return error_total / len(self.datos_x)
    
    def seleccion_torneo(self) -> Nodo:
        """
        Selecciona un individuo usando selección por torneo.
        
        Returns:
            Individuo seleccionado
        """
        torneo = random.sample(self.poblacion, self.tamaño_torneo)
        mejor = min(torneo, key=self.calcular_fitness)
        return mejor.copiar()
    
    def cruce_subtree(self, padre1: Nodo, padre2: Nodo) -> tuple[Nodo, Nodo]:
        """
        Realiza cruce de subárboles (subtree crossover) según Koza.
        Intercambia subárboles aleatorios entre dos padres.
        
        Args:
            padre1, padre2: Árboles padre
            
        Returns:
            Tupla con dos descendientes
        """
        hijo1 = padre1.copiar()
        hijo2 = padre2.copiar()
        
        # Seleccionar nodos aleatorios para intercambio
        nodos1 = self._obtener_todos_nodos(hijo1)
        nodos2 = self._obtener_todos_nodos(hijo2)
        
        if len(nodos1) > 1 and len(nodos2) > 1:
            # Evitar intercambiar la raíz siempre
            nodo1 = random.choice(nodos1[1:] if len(nodos1) > 1 else nodos1)
            nodo2 = random.choice(nodos2[1:] if len(nodos2) > 1 else nodos2)
            
            # Intercambiar subárboles
            nodo1.valor, nodo2.valor = nodo2.valor, nodo1.valor
            nodo1.hijos, nodo2.hijos = nodo2.hijos, nodo1.hijos
            nodo1.es_terminal, nodo2.es_terminal = nodo2.es_terminal, nodo1.es_terminal
        
        return hijo1, hijo2
    
    def mutacion_subtree(self, individuo: Nodo) -> Nodo:
        """
        Realiza mutación de subárbol (subtree mutation).
        Reemplaza un subárbol aleatorio con uno nuevo generado aleatoriamente.
        
        Args:
            individuo: Árbol a mutar
            
        Returns:
            Árbol mutado
        """
        mutado = individuo.copiar()
        nodos = self._obtener_todos_nodos(mutado)
        
        if len(nodos) > 1:
            # Seleccionar nodo aleatorio (evitar raíz si es posible)
            nodo_a_mutar = random.choice(nodos[1:] if len(nodos) > 1 else nodos)
            
            # Generar nuevo subárbol
            profundidad_restante = random.randint(1, 3)
            nuevo_subarbol = self.generar_arbol_aleatorio(profundidad_restante, 'grow')
            
            # Reemplazar
            nodo_a_mutar.valor = nuevo_subarbol.valor
            nodo_a_mutar.hijos = nuevo_subarbol.hijos
            nodo_a_mutar.es_terminal = nuevo_subarbol.es_terminal
        
        return mutado
    
    def _obtener_todos_nodos(self, raiz: Nodo) -> List[Nodo]:
        """
        Obtiene todos los nodos del árbol mediante recorrido en profundidad.
        
        Args:
            raiz: Nodo raíz del árbol
            
        Returns:
            Lista de todos los nodos
        """
        nodos = [raiz]
        for hijo in raiz.hijos:
            nodos.extend(self._obtener_todos_nodos(hijo))
        return nodos
    
    def evolucionar(self) -> Optional[Nodo]:
        """
        Ejecuta el algoritmo principal de Programación Genética.
        
        Returns:
            Mejor individuo encontrado o None si no se encuentra
        """
        print("Iniciando Programación Genética...")
        print(f"Función objetivo: y = x² + 2x + 1")
        print(f"Población: {self.tamaño_poblacion}, Generaciones: {self.max_generaciones}")
        print("-" * 60)
        
        # Inicializar población
        self.inicializar_poblacion()
        
        for generacion in range(self.max_generaciones):
            # Evaluar fitness de toda la población
            fitness_poblacion = [(individuo, self.calcular_fitness(individuo)) 
                               for individuo in self.poblacion]
            
            # Encontrar el mejor individuo de esta generación
            mejor_actual = min(fitness_poblacion, key=lambda x: x[1])
            
            if mejor_actual[1] < self.mejor_fitness:
                self.mejor_fitness = mejor_actual[1]
                self.mejor_individuo = mejor_actual[0].copiar()
            
            self.historial_fitness.append(self.mejor_fitness)
            
            # Mostrar progreso cada 10 generaciones
            if generacion % 10 == 0 or generacion == self.max_generaciones - 1:
                print(f"Generación {generacion:3d}: "
                      f"Mejor fitness = {self.mejor_fitness:.6f}, "
                      f"Mejor expresión = {self.mejor_individuo}")
            
            # Criterio de parada: fitness muy bueno
            if self.mejor_fitness < 0.01:
                print(f"\n¡Solución encontrada en generación {generacion}!")
                break
            
            # Crear nueva población
            nueva_poblacion = []
            
            # Elitismo: conservar el mejor individuo
            nueva_poblacion.append(self.mejor_individuo.copiar())
            
            # Generar resto de la población
            while len(nueva_poblacion) < self.tamaño_poblacion:
                if random.random() < self.prob_cruce:
                    # Cruce
                    padre1 = self.seleccion_torneo()
                    padre2 = self.seleccion_torneo()
                    hijo1, hijo2 = self.cruce_subtree(padre1, padre2)
                    
                    nueva_poblacion.extend([hijo1, hijo2])
                else:
                    # Reproducción directa
                    individuo = self.seleccion_torneo()
                    nueva_poblacion.append(individuo)
            
            # Aplicar mutación
            for i in range(1, len(nueva_poblacion)):  # Saltar el elite
                if random.random() < self.prob_mutacion:
                    nueva_poblacion[i] = self.mutacion_subtree(nueva_poblacion[i])
            
            # Truncar si es necesario
            self.poblacion = nueva_poblacion[:self.tamaño_poblacion]
        
        print("-" * 60)
        print(f"Evolución completada!")
        print(f"Mejor fitness final: {self.mejor_fitness:.6f}")
        print(f"Mejor expresión: {self.mejor_individuo}")
        
        return self.mejor_individuo
    
    def probar_mejor_individuo(self, individuo: Nodo, num_puntos: int = 10):
        """
        Prueba el mejor individuo en puntos específicos y muestra los resultados.
        
        Args:
            individuo: Mejor árbol encontrado
            num_puntos: Número de puntos de prueba
        """
        print("\nPrueba del mejor individuo:")
        print("x\t| Esperado\t| Predicho\t| Error")
        print("-" * 45)
        
        puntos_prueba = [random.uniform(-3, 3) for _ in range(num_puntos)]
        
        for x in sorted(puntos_prueba):
            y_esperado = x**2 + 2*x + 1
            y_predicho = individuo.evaluar(x)
            error = abs(y_esperado - y_predicho)
            
            print(f"{x:.2f}\t| {y_esperado:.4f}\t| {y_predicho:.4f}\t| {error:.4f}")

def main():
    """
    Función principal que ejecuta el ejemplo de Programación Genética.
    """
    print("=" * 70)
    print("PROGRAMACIÓN GENÉTICA - Ejemplo de John Koza")
    print("Evolución de expresiones matemáticas")
    print("=" * 70)
    
    # Configurar semilla para reproducibilidad (opcional)
    random.seed(42)
    
    # Crear instancia de PG con parámetros
    pg = ProgramacionGenetica(
        tamaño_poblacion=100,
        max_generaciones=50,
        profundidad_maxima=6,
        prob_cruce=0.9,
        prob_mutacion=0.1,
        tamaño_torneo=3
    )
    
    # Ejecutar evolución
    mejor_solucion = pg.evolucionar()
    
    # Probar la mejor solución
    if mejor_solucion is not None:
        pg.probar_mejor_individuo(mejor_solucion)
    else:
        print("No se encontró una solución válida.")
    
    print("\n" + "=" * 70)
    print("CONCEPTOS CLAVE DE LA PROGRAMACIÓN GENÉTICA:")
    print("=" * 70)
    print("• Representación: Árboles sintácticos (no cadenas binarias)")
    print("• Población inicial: Método 'ramped half-and-half'")
    print("• Selección: Torneo (mantiene diversidad)")
    print("• Cruce: Intercambio de subárboles")
    print("• Mutación: Reemplazo de subárboles")
    print("• Evaluación: Error en datos de entrenamiento")
    print("• Aplicaciones: Evolución de programas, funciones, etc.")
    print("=" * 70)

if __name__ == "__main__":
    main()