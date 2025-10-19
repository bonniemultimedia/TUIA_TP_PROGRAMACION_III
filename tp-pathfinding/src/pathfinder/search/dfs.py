from ..models.grid import Grid
from ..models.frontier import StackFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node



class DepthFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """Encuentra una ruta entre dos puntos en una cuadrícula usando Búsqueda en Profundidad

        Args:
            grid (Grid): Cuadrícula de puntos

        Returns:
            Solution: Solución encontrada
        """
        # Inicializar el nodo raíz
        nodo_raiz = Node("", state=grid.initial, cost=0, parent=None, action=None)

        # Verificar si el estado inicial es el objetivo
        if grid.objective_test(nodo_raiz.state):
            return Solution(nodo_raiz, {nodo_raiz.state: nodo_raiz})

        # Inicializar la frontera con el nodo raíz (usando pila)
        frontera = StackFrontier()
        frontera.add(nodo_raiz)

        # Inicializar el diccionario de nodos alcanzados con el nodo raíz
        alcanzados = {grid.initial: nodo_raiz}

        # Mientras la frontera no esté vacía
        while not frontera.is_empty():
            # Extraer el último nodo agregado (comportamiento LIFO de la pila)
            nodo_actual = frontera.remove()

            # Expandir el nodo actual: obtener todas las acciones posibles
            for accion in grid.actions(nodo_actual.state):
                # Calcular el nuevo estado resultante de aplicar la acción
                nuevo_estado = grid.result(nodo_actual.state, accion)
                
                # Verificar si el nuevo estado no ha sido alcanzado
                if nuevo_estado not in alcanzados:
                    # Calcular el nuevo costo acumulado
                    nuevo_costo = nodo_actual.cost + grid.individual_cost(nodo_actual.state, accion)
                    
                    # Crear un nuevo nodo
                    nuevo_nodo = Node("", nuevo_estado, nuevo_costo, nodo_actual, accion)

                    # Verificar si el nuevo estado es el objetivo
                    if grid.objective_test(nuevo_estado):
                        # Agregar el nuevo nodo al diccionario de alcanzados y retornar solución
                        alcanzados[nuevo_estado] = nuevo_nodo
                        return Solution(nuevo_nodo, alcanzados)
                    
                    # Agregar el nuevo nodo a la frontera y al diccionario de alcanzados
                    frontera.add(nuevo_nodo)
                    alcanzados[nuevo_estado] = nuevo_nodo

        # Retornar que no se encontró solución, con los nodos alcanzados hasta el momento
        return NoSolution(alcanzados)