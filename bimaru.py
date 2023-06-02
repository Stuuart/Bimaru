# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# 4Boat / four_boat => Barco de tamanho 4
# 3Boat / three_boat => Barco de tamanho 3
# 2Boat / two_boat => Barco de tamanho 2
# 1Boat / one_boat => Barco de tamanho 1

# Grupo 152:
# 96196 Duarte Manuel da Cruz Costa
# 102492 Francisco Dias Garcia da Fonseca

from sys import stdin
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)
import numpy as np

class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = Board()
        self.board.row_elements = board.row_elements
        self.board.col_elements = board.col_elements
        self.board.game_cells = np.copy(board.game_cells)   # Avoids changing previous boards
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self):
        self.row_elements = []
        self.col_elements = []
        self.game_cells = np.empty((10, 10), dtype=np.chararray)

        # TODO Best way to do it?
        self.one_boat = 4
        self.two_boat = 3
        self.three_boat = 2
        self.four_boat = 1
    
    def __str__(self):
        string_representation = ''.join((str(row)) for row in self.row_elements) + '\n' + ''.join((str(col)) for col in self.col_elements) + '\n'
        string_representation += '\n'.join([''.join(row.astype(str)) for row in self.game_cells])
        return string_representation.replace('None', '_')

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.game_cells[row, col]

    def set_value(self, row: int, col: int, val: int):
        self.game_cells[row, col] = val

    # recebe coordenada da column, devolve quantas pecas ainda podem ser postas 
    def check_col_parts(self, col:int):
        parts = np.count_nonzero(self.game_cells[:, col])
        water = np.count_nonzero(np.logical_or(self.game_cells[:, col] == 'W', self.game_cells[:, col] == '.'))
        return int(self.col_elements[col]) - parts + water

    # recebe coordenada da row, devolve quantas pecas ainda podem ser postas 
    def check_row_parts(self, row:int):
        parts = np.count_nonzero(self.game_cells[row, :])
        water = np.count_nonzero(np.logical_or(self.game_cells[row, :] == 'W', self.game_cells[row, :] == '.'))
        return int(self.row_elements[row]) - parts + water

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        down_value = "Out"
        up_value = "Out"

        if row < 9 and row > 0:
            down_value = self.game_cells[row + 1, col]
            up_value = self.game_cells[row - 1, col]
        elif row == 9:
            up_value = self.game_cells[row - 1, col]
        elif row == 0:
            down_value = self.game_cells[row + 1, col]
        return up_value, down_value

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""

        right_value = "Out"
        left_value = "Out"

        if col < 9 and col > 0:
            right_value = self.game_cells[row, col + 1]
            left_value = self.game_cells[row, col - 1]
        elif col == 9:
            left_value = self.game_cells[row, col - 1]
        elif col == 0:
            right_value = self.game_cells[row, col + 1]
        return left_value, right_value

    def adjacent_diagonal_values(self, row: int, col: int) -> (str, str, str, str):
        """Devolve os valores das diagonais adjacentes da célula"""

        l_up = "Out"
        l_down = "Out"
        r_up = "Out"
        r_down = "Out"

        if col == 0:
            if row > 0 and row < 9:
                r_up = self.game_cells[row - 1, col + 1]
                r_down = self.game_cells[row + 1, col + 1]
            elif row == 0:
                r_down = self.game_cells[row + 1, col + 1]
            elif row == 9:
                r_up = self.game_cells[row - 1, col + 1]
        elif col == 9:
            if row > 0 and row < 9:
                l_up = self.game_cells[row - 1, col - 1]
                l_down = self.game_cells[row + 1, col - 1]
            elif row == 0:
                l_down = self.game_cells[row + 1, col - 1]
            elif row == 9:
                l_up = self.game_cells[row - 1, col - 1]
        else:
            if row > 0 and row < 9:
                r_up = self.game_cells[row - 1, col + 1]
                r_down = self.game_cells[row + 1, col + 1]
                l_up = self.game_cells[row - 1, col - 1]
                l_down = self.game_cells[row + 1, col - 1]
            elif row == 0:
                r_down = self.game_cells[row + 1, col + 1]
                l_down = self.game_cells[row + 1, col - 1]
            elif row == 9:
                r_up = self.game_cells[row - 1, col + 1]
                l_up = self.game_cells[row - 1, col - 1]

        return r_up, r_down, l_up, l_down

    # Returns a list of coordinates for a specified element in the board
    def find(self, element: str):
        return np.argwhere(self.game_cells == element) 

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        initial_board = Board()

        # Gets the number of boats elements per column/row
        for i in range(2):
            line = stdin.readline().split()
            if line[0] == 'ROW':
                initial_board.row_elements = line[1:]
            elif line[0] == 'COLUMN':
                initial_board.col_elements = line[1:]
            else:
                print("Row or Column info missing!")
                return None
        
        # Reads number of hints given
        line = stdin.readline().split()
        if len(line) != 1:
            print("NO HINTS!")
            return None
        num_hints = int(line[0])

        # Writes the hints in the board
        for i in range(num_hints):
            line = stdin.readline().split()
            if line[0] == "HINT":
                initial_board.game_cells[int(line[1]), int(line[2])] = line[3]
            else:
                print("Problem with HINT!")
                return None

        initial_board.water_fill()
        return initial_board

    # TODO: outros metodos da classe

    # Used to fill with water in columns/lines where no more boats can be put
    def water_fill(self):
        
        # Surround C/c's with water
        indices = self.find('C')
        np.append(indices, self.find('c'))
        
        if len(indices) != 0:
            for index in indices:
                x = index[0]
                y = index[1]
                (left, right) = self.adjacent_horizontal_values(x,y)
                (up, down) = self.adjacent_vertical_values(x,y)
                (r_up, r_down, l_up, l_down) = self.adjacent_diagonal_values(x,y)

                if l_up != ('W' and '.' and "Out"):
                    self.set_value(x-1, y-1, '.')
                if r_up != ('W' and '.' and "Out"):
                    self.set_value(x-1, y+1, '.')

                if l_down != ('W' and '.' and "Out"):
                    self.set_value(x+1, y-1, '.')
                if r_down != ('W' and '.' and "Out"):
                    self.set_value(x+1, y+1, '.')

                if up != ('W' and '.' and "Out"):
                    self.set_value(x-1, y, '.')
                if down != ('W' and '.' and "Out"):
                    self.set_value(x+1, y, '.')

                if left != ('W' and '.' and "Out"):
                    self.set_value(x, y-1, '.')
                if right != ('W' and '.' and "Out"):
                    self.set_value(x, y+1, '.')

        # for lines
        for i in range(10):
            # for row
            filled_cells = np.count_nonzero(self.game_cells[i, :])
            elements = int(self.row_elements[i])
            water_tips = np.count_nonzero(np.logical_or(self.game_cells[i, :] == 'W', self.game_cells[i, :] == '.'))
            if filled_cells - elements - water_tips == 0:
                self.game_cells[i, :] = ['.' if cell is None else cell for cell in self.game_cells[i, :]]

            # for column
            filled_cells = np.count_nonzero(self.game_cells[:, i])
            elements = int(self.col_elements[i])
            water_tips = np.count_nonzero(np.logical_or(self.game_cells[:, i] == 'W', self.game_cells[:, i] == '.'))
            if filled_cells - elements - water_tips == 0:
                self.game_cells[:, i] = ['.' if cell is None else cell for cell in self.game_cells[:, i]]
        

    def check_surroundings(self, coords, size: int):
        x = coords[0]
        y = coords[1]
        boats = ['T', 'B', 'L', 'R', 'M', 'C', 't', 'b', 'l', 'r', 'm', 'c']
        surround = []
        surround.extend(self.adjacent_horizontal_values(x,y))
        surround.extend(self.adjacent_vertical_values(x,y))
        surround.extend(self.adjacent_diagonal_values(x,y))

        return any(s in boats for s in surround)

    # Este size eh igual a size-1 no metodo occurs
    def find_in_distance(self, origin, size: int, direction: str):

        x = origin[0]
        y = origin[1]

        if self.check_surroundings((x,y), size):
            return None

        val = self.get_value(x,y)         
        boat = [[x,y,val]]
        acpt = [None, 'M', 'm']
        for i in range(size):
            
            if direction == "up":
                if x < 0 + size: return None
                if (self.check_row_parts(x-i-1) < 1) or (self.check_col_parts(y) < size-i): return None
                if val == None: boat[0][2] = 'b'
                boat.append([x-i-1, y, self.adjacent_vertical_values(x-i-1,y)[0]])
                if i == size-1: acpt = [None, 'T', 't']
            elif direction == "down":
                if x > 9 - size: return None
                if (self.check_row_parts(x+i+1) < 1) or (self.check_col_parts(y) < size-i): return None
                if val == None: boat[0][2] = 't'
                boat.append([x+i+1, y, self.adjacent_vertical_values(x+i+1,y)[1]])                
                if i == size-1: acpt = [None, 'B', 'b']
            elif direction == "left":
                if y < 0 + size: return None
                if (self.check_row_parts(x) < 1) or (self.check_col_parts(y-i-1) < size-i): return None
                if val == None: boat[0][2] = 'r'
                boat.append([x, y-i-1, self.adjacent_horizontal_values(x,y-i-1)[0]])
                if i == size-1: acpt = [None, 'L', 'l']
            elif direction == "right":
                if y > 9 - size: return None
                if (self.check_row_parts(x) < 1) or (self.check_col_parts(y+i+1) < size-i): return None
                if val == None: boat[0][2] = 'l'
                boat.append([x, y+i+1, self.adjacent_horizontal_values(x,y+i+1)[1]])
                if i == size-1: acpt = [None, 'R', 'r']
            else:
                return None

            if boat[i+1][2] in acpt:
                if boat[i+1][2] == None: boat[i+1][2] = acpt[2]
                continue
            else:
                return None

        return boat
        
    def occurs(self, element: str, size: int, direction: str):
        boats = []
        indices = self.find(element) 
        if element != None:
            np.append(indices, self.find(element.casefold()))
        if len(indices) != 0:
            for index in indices:
                x = index[0]
                y = index[1]
                val = self.get_value(x, y)

                boat = self.find_in_distance([x,y], size-1, direction)           
                if boat != None:
                    boats.append(boat)
                     
        return boats

    # Encontrar 4 quadrados consecutivos ou usar partes oferecidas pelas HINTs
    # Recebe um BimaruState e devolve uma lista com as possiveis posicoes do 4Boat
    # Cada possivel posicao do 4Boat eh em si uma lista de coordenadas
    def find_4Boat_actions(self):
        
        possible_actions = []

        # Find indices where 'T' or 't' occurs
        possible_actions.extend(self.occurs('T', 4, "down"))
        # Find indices where 'B' or 'b' occurs
        possible_actions.extend(self.occurs('B', 4, "up"))
        # Find indices where 'L' or 'l' occurs
        possible_actions.extend(self.occurs('L', 4, "right"))
        # Find indices where 'R' or 'r' occurs
        possible_actions.extend(self.occurs('R', 4, "left"))

        # Find empty spaces and treat as extremities
        possible_actions.extend(self.occurs(None, 4, "down"))
        possible_actions.extend(self.occurs(None, 4, "up"))
        possible_actions.extend(self.occurs(None, 4, "right"))
        possible_actions.extend(self.occurs(None, 4, "left"))

        return possible_actions
    
    # Encontrar 3 quadrados consecutivos ou usar partes oferecidas pelas HINTs
    # Recebe um BimaruState e devolve uma lista com as possiveis posicoes do 4Boat
    # Cada possivel posicao do 3Boat eh em si uma lista de coordenadas
    def find_3Boat_actions(self):
        
        possible_actions = []

        # Find indices where 'T' or 't' occurs
        possible_actions.extend(self.occurs('T', 3, "down"))
        # Find indices where 'B' or 'b' occurs
        possible_actions.extend(self.occurs('B', 3, "up"))
        # Find indices where 'L' or 'l' occurs
        possible_actions.extend(self.occurs('L', 3, "right"))
        # Find indices where 'R' or 'r' occurs
        possible_actions.extend(self.occurs('R', 3, "left"))

        # Find empty spaces and treat as extremities
        possible_actions.extend(self.occurs(None, 3, "down"))
        possible_actions.extend(self.occurs(None, 3, "up"))
        possible_actions.extend(self.occurs(None, 3, "right"))
        possible_actions.extend(self.occurs(None, 3, "left"))

        return possible_actions

    # Encontrar 3 quadrados consecutivos ou usar partes oferecidas pelas HINTs
    # Recebe um BimaruState e devolve uma lista com as possiveis posicoes do 4Boat
    # Cada possivel posicao do 3Boat eh em si uma lista de coordenadas
    def find_2Boat_actions(self):
        
        possible_actions = []

        # Find indices where 'T' or 't' occurs
        possible_actions.extend(self.occurs('T', 2, "down"))
        # Find indices where 'B' or 'b' occurs
        possible_actions.extend(self.occurs('B', 2, "up"))
        # Find indices where 'L' or 'l' occurs
        possible_actions.extend(self.occurs('L', 2, "right"))
        # Find indices where 'R' or 'r' occurs
        possible_actions.extend(self.occurs('R', 2, "left"))

        # Find empty spaces and treat as extremities
        possible_actions.extend(self.occurs(None, 2, "down"))
        possible_actions.extend(self.occurs(None, 2, "up"))
        possible_actions.extend(self.occurs(None, 2, "right"))
        possible_actions.extend(self.occurs(None, 2, "left"))

        return possible_actions
    


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO Confuso af

    
        # Actions para 4_boat, depois result, depois actions para 3_boat, repeat...
        #possible_actions = []

        if state.board.four_boat != 0:
           return state.board.find_4Boat_actions()
        
        #if state.board.three_boat != 0:
         #   possible_actions.append(find_3Boat_actions(state))

        #if state.board.two_boat != 0:
         #   possible_actions.append(find_2Boat_actions(state))
        
        #if state.board.one_boat != 0:
         #   possible_actions.append(find_1Boat_actions(state))
    
        # Find 3 consecutive squares in a row or col
        # or use boat parts given by hints
    
        # Find 2 consecutive squares in a row or col
        # or use boat parts given by hints

        # Find 1 free square
        return possible_actions


    # action: list of list of int/char
    # Place a boat, surrounded by water
    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO

        new_state = BimaruState(state.board)

        for a in action:
            new_state.board.set_value(a[0], a[1], a[2])

        # TODO fill surrounding with water
        """if action[0][2] == 'b':
            new_state.board.game_cells[action[0][0] + 1, action[0][1]] = '.'
            new_state.board.game_cells[action[0][0] + 1, action[0][1] + 1] = '.'"""

        new_state.board.water_fill()

        # When four_boat is placed, state.board.four_boat -= 1

        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO

        empty_spaces = state.board.game_cells.size - np.count_nonzero(state.board.game_cells)

        one_boat = state.board.one_boat
        two_boat = state.board.two_boat
        three_boat = state.board.three_boat
        four_boat = state.board.four_boat

        return (one_boat+two_boat+three_boat+four_boat) + empty_spaces == 0

        """for i in range(10):
            state.board.game_cells
            print(state.board.game_cells)
            if self.count_pieces(state.board.game_cells, i, False) != state.board.row_elements[i]:
                return False
            if self.count_pieces(state.board.game_cells, i, True) != state.board.col_elements[i]:
                return False
        if state.board.game_cells.size - np.count_nonzero(state.board.game_cells) != 0:
            return False

        return True"""

    """@staticmethod
    def count_pieces(l, num, rc):
        val = 0
        if rc:
            for i in range(10):
                if (not (l[num][i] == '.' or l[num][i] is None or l[num][i] == 'W')):
                    val += 1
        else:
            for i in range(10):
                if (not (l[i][num] == '.' or l[i][num] is None or l[i][num] == 'W')):
                    val += 1
        return val"""

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass
    
    # TODO: outros metodos da classe

if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    game_board = Board.parse_instance()

    problem = Bimaru(game_board)

    #goal_node = depth_first_tree_search(problem)    

    actions = problem.actions(problem.initial)
    print(actions)

    #state1 = problem.result(problem.initial, )

    #print(goal_node)

    # Convert array to string where each row is a line

    print(problem.initial.board)

    pass
