from ..models.grid import Grid
from ..models.frontier import PriorityQueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node


class GreedyBestFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """Encuentra un camino entre dos puntos en una cuadrícula usando Búsqueda Voraz Primero el Mejor

        Args:
            grid (Grid): Cuadrícula de puntos

        Returns:
            Solution: Solución encontrada
        """
        # Inicializar el nodo raíz
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)

        # Inicializar alcanzados con el estado inicial
        reached = {}
        reached[root.state] = root.cost

        # Inicializar la frontera con el nodo raíz usando la heurística
        frontier = PriorityQueueFrontier()
        frontier.add(root, GreedyBestFirstSearch.heuristic(root.state, grid.end))

        # Mientras la frontera no esté vacía
        while not frontier.is_empty():
            # Extraer el nodo con menor valor heurístico
            node = frontier.pop()

            # Verificar si el nodo actual es el objetivo
            if grid.objective_test(node.state):
                return Solution(node, reached)

            # Expandir el nodo actual
            for action in grid.actions(node.state):
                # Calcular el nuevo estado
                new_state = grid.result(node.state, action)
                
                # Calcular el nuevo costo
                new_cost = node.cost + grid.individual_cost(node.state, action)

                # Si el nuevo estado no ha sido alcanzado o tiene menor costo
                if new_state not in reached or new_cost < reached[new_state]:
                    # Crear nuevo nodo
                    new_node = Node("", new_state, new_cost, node, action)
                    
                    # Actualizar el diccionario de alcanzados
                    reached[new_state] = new_cost
                    
                    # Calcular la heurística para el nuevo estado
                    h_value = GreedyBestFirstSearch.heuristic(new_state, grid.end)
                    
                    # Agregar el nuevo nodo a la frontera con prioridad heurística
                    frontier.add(new_node, h_value)

        return NoSolution(reached)

    @staticmethod
    def heuristic(state: tuple[int, int], goal: tuple[int, int]) -> int:
        """Calcula la heurística de distancia Manhattan entre un estado y el objetivo
        
        Args:
            state (tuple): Estado actual (fila, columna)
            goal (tuple): Estado objetivo (fila, columna)
            
        Returns:
            int: Valor heurístico
        """
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])
