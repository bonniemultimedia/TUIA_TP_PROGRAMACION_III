from ..models.grid import Grid
from ..models.frontier import PriorityQueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node


class AStarSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """Encuentra una ruta entre dos puntos en una cuadrícula usando Búsqueda A*

        Args:
            grid (Grid): Cuadrícula de puntos

        Returns:
            Solution: Solución encontrada
        """
        # Inicializa el nodo raíz
        nodo_raiz = Node("", state=grid.initial, cost=0, parent=None, action=None)

        # Inicializa el diccionario de estados alcanzados con el estado inicial y su costo
        estados_alcanzados = {}
        estados_alcanzados[nodo_raiz.state] = nodo_raiz.cost

        # Inicializa la frontera con el nodo raíz
        frontera = PriorityQueueFrontier()
        
        # Calcula la heurística para el nodo raíz (distancia Manhattan al objetivo)
        heuristica_raiz = abs(nodo_raiz.state[0] - grid.end[0]) + abs(nodo_raiz.state[1] - grid.end[1])
        costo_total_raiz = nodo_raiz.cost + heuristica_raiz
        frontera.add(nodo_raiz, costo_total_raiz)

        # Bucle principal de búsqueda
        while not frontera.is_empty():
            # Extrae el nodo con menor f(n) de la frontera usando pop
            nodo_actual = frontera.pop()

            # Verifica si el nodo actual es el objetivo
            if grid.objective_test(nodo_actual.state):
                return Solution(nodo_actual, estados_alcanzados)

            # Expande el nodo actual: para cada acción posible
            for accion in grid.actions(nodo_actual.state):
                # Calcula el nuevo estado después de aplicar la acción
                nuevo_estado = grid.result(nodo_actual.state, accion)

                # Calcula el nuevo costo g(n) = costo acumulado desde el inicio
                nuevo_costo_g = nodo_actual.cost + grid.individual_cost(nodo_actual.state, accion)

                # Si el nuevo estado no ha sido alcanzado o tiene un costo g menor
                if nuevo_estado not in estados_alcanzados or nuevo_costo_g < estados_alcanzados[nuevo_estado]:
                    # Actualiza el diccionario de estados alcanzados con el nuevo costo g
                    estados_alcanzados[nuevo_estado] = nuevo_costo_g

                    # Crea un nuevo nodo
                    nuevo_nodo = Node(
                        value="",
                        state=nuevo_estado,
                        cost=nuevo_costo_g,
                        parent=nodo_actual,
                        action=accion
                    )

                    # Calcula la heurística para el nuevo nodo (distancia Manhattan al objetivo)
                    heuristica_nueva = abs(nuevo_estado[0] - grid.end[0]) + abs(nuevo_estado[1] - grid.end[1])
                    
                    # Calcula f(n) = g(n) + h(n) para el nuevo nodo
                    costo_total = nuevo_costo_g + heuristica_nueva

                    # Añade el nuevo nodo a la frontera con f(n) como prioridad
                    frontera.add(nuevo_nodo, costo_total)

        # No se encontró solución
        return NoSolution(estados_alcanzados)