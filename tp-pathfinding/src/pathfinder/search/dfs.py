from ..models.grid import Grid
from ..models.frontier import StackFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node



class DepthFirstSearch:
    @staticmethod
    def search(grid) -> Solution:
        """Find path between two points in a dfs using Depth First Search

        Args:
            dfs (DepthFirstSearch): DepthFirstSearch of points

        Returns:
            Solution: Solution found
        """
         # Inicializar el nodo raíz
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)

        # Verificar si el estado inicial es el objetivo
        if grid.objective_test(root.state):
            return Solution(root, {root.state: root})

        # Inicializar la frontera con el nodo raíz (usando pila)
        frontier = StackFrontier()
        frontier.add(root)

        # Inicializar el diccionario de nodos alcanzados con el nodo raíz
        reached = {grid.initial: root}

        # Mientras la frontera no esté vacía
        while not frontier.is_empty():

            # Extraer el último nodo agregado (comportamiento LIFO de la pila)
            node = frontier.remove()

            # Expandir el nodo actual: obtener todas las acciones posibles
            for action in grid.actions(node.state):
                # Calcular el nuevo estado resultante de aplicar la acción
                new_state = grid.result(node.state, action)
                
                # Calcular el nuevo costo acumulado
                new_cost = node.cost + grid.individual_cost(node.state, action)
                
                # Crear un nuevo nodo
                new_node = Node("", new_state, new_cost, node, action)

                # Verificar si hay un ciclo en la ruta desde el nuevo nodo hasta la raíz
                if not DepthFirstSearch.has_cycle(new_state, node):
                    # Verificar si el nuevo estado es el objetivo
                    if grid.objective_test(new_state):
                        # Agregar el nuevo nodo al diccionario de alcanzados y retornar solución
                        reached[new_state] = new_node
                        return Solution(new_node, reached)
                    
                    # Agregar el nuevo nodo a la frontera y al diccionario de alcanzados
                    frontier.add(new_node)
                    reached[new_state] = new_node

        # Retornar que no se encontró solución, con los nodos alcanzados hasta el momento
        return NoSolution(reached)

    @staticmethod
    def has_cycle(state: tuple[int, int], node: Node) -> bool:
        """Verifica si hay un ciclo en la ruta desde el nodo dado hasta la raíz

        Args:
            state (tuple): Estado a verificar
            node (Node): Nodo desde el cual comenzar la verificación hacia la raíz

        Returns:
            bool: True si hay ciclo, False en caso contrario
        """
        current = node
        while current is not None:
            if current.state == state:
                return True
            current = current.parent
        return False