from ..models.grid import Grid
from ..models.frontier import QueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node


class BreadthFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """Find path between two points in a grid using Breadth First Search

        Args:
            grid (Grid): Grid of points

        Returns:
            Solution: Solution found
        """
        # Inicializar el nodo raíz
        root = Node("", state=grid.initial, cost=0, parent=None, action=None) 
        
        # Inicializar el diccionario de estados alcanzados con el estado inicial
        reached = {} 
        reached[root.state] = True 

         # Verificar si el estado inicial es la solución
        if grid.objective_test(root.state):
            return Solution(root, reached)
            
         # Inicializar la frontera y agregar el nodo raíz
        frontier = QueueFrontier()
        frontier.add(root)

        # Mientras la frontera no esté vacía
        while not frontier.is_empty():

            # Remover un nodo de la frontera
            node = frontier.remove()
            
            # Obtener todas las acciones posibles desde el estado actual
            actions = grid.actions(node.state)

            # Expandir el nodo para cada acción posible
            for action in actions:

                # Obtener el estado resultante de aplicar la acción
                new_state = grid.result(node.state, action)
                
                # Verificar si el nuevo estado es la solución
                if grid.objective_test(new_state):
                    
                    # Crear el nodo solución
                    new_node = Node("", new_state, 
                                   node.cost + grid.individual_cost(node.state, action),
                                   node, action)
                    reached[new_state] = True
                    return Solution(new_node, reached)

                # Si el nuevo estado no ha sido alcanzado, agregarlo a la frontera
                if new_state not in reached:
                    reached[new_state] = True
                    new_node = Node("", new_state, 
                                   node.cost + grid.individual_cost(node.state, action),
                                   node, action)
                    frontier.add(new_node)    
        
        # Retornar solución vacía si no se encontró camino
        return NoSolution(reached)
