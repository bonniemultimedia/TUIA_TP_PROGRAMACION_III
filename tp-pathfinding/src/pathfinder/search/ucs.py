from ..models.grid import Grid
from ..models.frontier import PriorityQueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node


class UniformCostSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """Encuentra una ruta entre dos puntos en una cuadrícula usando Búsqueda de Costo Uniforme

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

        # Inicializa la frontera con el nodo raíz usando su costo como prioridad
        frontera = PriorityQueueFrontier()
        frontera.add(nodo_raiz, nodo_raiz.cost)

        # Bucle principal de búsqueda
        while not frontera.is_empty():
            # Extrae el nodo con menor costo de la frontera
            nodo_actual = frontera.pop()

            # Verifica si el nodo actual es el objetivo
            if grid.objective_test(nodo_actual.state):
                return Solution(nodo_actual, estados_alcanzados)

            # Expande el nodo actual: para cada acción posible
            for accion in grid.actions(nodo_actual.state):
                # Calcula el nuevo estado después de aplicar la acción
                nuevo_estado = grid.result(nodo_actual.state, accion)

                # Calcula el nuevo costo acumulado
                nuevo_costo = nodo_actual.cost + grid.individual_cost(nodo_actual.state, accion)

                # Si el nuevo estado no ha sido alcanzado o tiene un costo menor
                if nuevo_estado not in estados_alcanzados or nuevo_costo < estados_alcanzados[nuevo_estado]:
                    # Actualiza el diccionario de estados alcanzados
                    estados_alcanzados[nuevo_estado] = nuevo_costo

                    # Crea un nuevo nodo
                    nuevo_nodo = Node(
                        value="",
                        state=nuevo_estado,
                        cost=nuevo_costo,
                        parent=nodo_actual,
                        action=accion
                    )

                    # Añade el nuevo nodo a la frontera con su costo como prioridad
                    frontera.add(nuevo_nodo, nuevo_costo)

        # No se encontró solución
        return NoSolution(estados_alcanzados)