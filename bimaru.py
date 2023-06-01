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
        return self.row_elements[col] - parts + water

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
                r_down = self.game_cells[row + 1, col - 1]
            elif row == 0:
                r_down = self.game_cells[row + 1, col - 1]
            elif row == 9:
                r_up = self.game_cells[row - 1, col + 1]
        elif col == 9:
            if row > 0 and row < 9:
                l_up = self.game_cells[row - 1, col + 1]
                l_down = self.game_cells[row + 1, col + 1]
            elif row == 0:
                l_down = self.game_cells[row + 1, col + 1]
            elif row == 9:
                l_up = self.game_cells[row - 1, col + 1]
        else:
            if row > 0 and row < 9:
                r_up = self.game_cells[row - 1, col + 1]
                r_down = self.game_cells[row + 1, col - 1]
                l_up = self.game_cells[row - 1, col + 1]
                l_down = self.game_cells[row + 1, col + 1]
            elif row == 0:
                r_down = self.game_cells[row + 1, col - 1]
                l_down = self.game_cells[row + 1, col + 1]
            elif row == 9:
                r_up = self.game_cells[row - 1, col + 1]
                l_up = self.game_cells[row - 1, col + 1]

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
        
    # Recebe duas coordenadas, c1 e c2, e um tuplo de elementos que pode aceitar, acpt,
    # devolve True se apenas existirem elementos de e entre c1 e c2
    def find_between(self, c1, c2, acpt):
        elementos = []
        for i in range(len(acpt)):
            np.append(elementos, self.find(acpt[i]))

    # Encontrar 4 quadrados consecutivos ou usar partes oferecidas pelas HINTs
    # Recebe um BimaruState e devolve uma lista com as possiveis posicoes do 4Boat
    # Cada possivel posicao do 4Boat eh em si uma lista de coordenadas
    def find_4Boat_actions(self):
        
        possible_actions = []

        # Find indices where 'T' or 't' occurs
        indices = self.find('T') 
        np.append(indices, self.find('t'))
        if len(indices) != 0:
            for index in indices:
                x = index[0]
                y = index[1]
                val = self.get_value(x,y)

                # Check if there is space in the column for a 4Boat
                if self.check_col_parts(y) < 3: continue

                down = self.adjacent_vertical_values(x,y)[1]
                down2 = self.adjacent_vertical_values(x+1,y)[1]
                down3 = self.adjacent_vertical_values(x+2,y)[1]
                
                # If the middle spaces aren't None, 'M' or 'm', it isn't a possible action
                if down3 != (None and 'B' and 'b') or (down or down2) != (None and 'M' and 'm'): continue
    
                if down != 'M': down = 'm'
                if down2 != 'M': down2 = 'm'
                if down3 != 'B': down3 = 'b'

                possible_actions.append([[x, y, val], [x+1, y, down], [x+2, y, down2], [x+3, y, down3]])

        # Find indices where 'B' or 'b' occurs
        indices = self.find('B') 
        np.append(indices, self.find('b'))
        if len(indices) != 0:
            for index in indices:
                x = index[0]
                y = index[1]
                val = self.get_value(x,y)

                # Check if there is space in the column for a 4Boat
                if self.check_col_parts(y) < 3: continue

                up = self.adjacent_vertical_values(x,y)[0]
                up2 = self.adjacent_vertical_values(x-1,y)[0]
                up3 = self.adjacent_vertical_values(x-2,y)[0]
                
                # If the middle spaces aren't None, 'M' or 'm', it isn't a possible action
                if up3 != (None and 'T' and 't') or (up or up2) != (None and 'M' and 'm'): continue

                if up != 'M': up = 'm'
                if up2 != 'M': up2 = 'm'
                if up3 != 'T': up3 = 't'

                possible_actions.append([[x-3, y, up3], [x-2, y, up2], [x-1, y, up], [x, y, val]])

        # Find indices where 'L' or 'l' occurs
        indices = self.find('L') 
        np.append(indices, self.find('l'))
        if len(indices) != 0:
            for index in indices:
                x = index[0]
                y = index[1]
                val = self.get_value(x,y)

                # Check if there is space in the column for a 4Boat
                if self.check_col_parts(y) < 3: continue

                right = self.adjacent_horizontal_values(x,y)[1]
                right2 = self.adjacent_horizontal_values(x,y+1)[1]
                right3 = self.adjacent_horizontal_values(x,y+2)[1]
                
                # If the middle spaces aren't None, 'M' or 'm', it isn't a possible action
                if right3 != (None and 'R' and 'r') or (right or right2) != (None and 'M' and 'm'): continue

                if right != 'M': right = 'm'
                if right2 != 'M': right2 = 'm'
                if right3 != 'T': right3 = 't'

                possible_actions.append([[x, y, val], [x, y+1, right], [x, y+2, right2], [x, y+3, right3]])

        # Find indices where 'R' or 'r' occurs
        indices = self.find('R') 
        np.append(indices, self.find('r'))
        if len(indices) != 0:
            for index in indices:
                x = index[0]
                y = index[1]
                val = self.get_value(x,y)

                # Check if there is space in the column for a 4Boat
                if self.check_col_parts(y) < 3: continue

                left = self.adjacent_horizontal_values(x,y)[0]
                left2 = self.adjacent_horizontal_values(x,y-1)[0]
                left3 = self.adjacent_horizontal_values(x,y-2)[0]
                
                # If the middle spaces aren't None, 'M' or 'm', it isn't a possible action
                if left3 != (None and 'L' and 'l') or (left or left2) != (None and 'M' and 'm'): continue

                if left != 'M': left = 'm'
                if left2 != 'M': left2 = 'm'
                if left3 != 'T': left3 = 't'

                possible_actions.append([[x, y-3, left3], [x, y-2, left2], [x, y-1, left], [x, y, val]])

        # Find indices where 'M' or 'm' occurs
        indices = self.find('M') 
        np.append(indices, self.find('m'))
        if len(indices) != 0:
            for index in indices:
                x = index[0]
                y = index[1]
                val = self.get_value(x,y)

                # Check if there is space in the column for a 4Boat
                print(self.check_col_parts(x))
                if self.check_col_parts(y) < 3: continue
                if self.check_row_parts(y) < 3: continue

                left = self.adjacent_horizontal_values(x,y)[0]
                left2 = self.adjacent_vertical_values(x,y-1)[0]

                right = self.adjacent_horizontal_values(x,y)[1]
                right2 = self.adjacent_vertical_values(x,y+1)[1]
                # 2Left 1Right 
                for i in range(1):
                    if right != (None and 'R' and 'r'): continue
                    if left != (None and 'M' and 'm'): continue
                    if left2 != (None and 'L' and 'l'): continue

                    if right != 'R': right = 'r'
                    if left != 'M': left = 'm'
                    if left2 != 'L': left2 = 'l'
                    possible_actions.append([[x, y-2, left2], [x, y-1, left], [x, y, val], [x, y+1, right]])

                # 1Left 2Right 
                for i in range(1):
                    if left != (None and 'L' and 'l'): continue
                    if right != (None and 'M' and 'm'): continue
                    if right2 != (None and 'R' and 'r'): continue

                    if left != 'L': left = 'l'
                    if right != 'M': right = 'm'
                    if right2 != 'R': right2 = 'r'
                    possible_actions.append([[x, y-1, left], [x, y, val], [x, y+1, right], [x, y+2, right2]])

                up = self.adjacent_horizontal_values(x,y)[0]
                up2 = self.adjacent_vertical_values(x-1,y)[0]

                down = self.adjacent_horizontal_values(x,y)[1]
                down2 = self.adjacent_vertical_values(x+1,y)[1]
                # 2Up 1Down 
                for i in range(1):
                    if down != (None and 'B' and 'b'): continue
                    if up != (None and 'M' and 'm'): continue
                    if up2 != (None and 'T' and 't'): continue

                    if down != 'B': left = 'b'
                    if up != 'M': up = 'm'
                    if up2 != 'T': left3 = 't'
                    possible_actions.append([[x-2, y, up2], [x-1, y, up], [x, y, val], [x+1, y, down]])

                # 1Up 2Down 
                for i in range(1):
                    if up != (None and 'T' and 't'): continue
                    if down != (None and 'M' and 'm'): continue
                    if down2 != (None and 'B' and 'b'): continue

                    if up != 'T': up = 't'
                    if down != 'M': left = 'm'
                    if down2 != 'B': left3 = 'b'
                    possible_actions.append([[x-1, y, up], [x, y, val], [x+1, y, down], [x+2, y, down2]])

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

        print(self.actions(state))
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

        """one_boat = state.board.one_boat
        two_boat = state.board.two_boat
        three_boat = state.board.three_boat
        four_boat = state.board.four_boat

        return (one_boat+two_boat+three_boat+four_boat) == 0"""

        for i in range(10):
            state.board.game_cells
            print(state.board.game_cells)
            if self.count_pieces(state.board.game_cells, i, False) != state.board.row_elements[i]:
                return False
            if self.count_pieces(state.board.game_cells, i, True) != state.board.col_elements[i]:
                return False
        if state.board.game_cells.size - np.count_nonzero(state.board.game_cells) != 0:
            return False

        return True

    @staticmethod
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
        return val

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass
    
    # TODO: outros metodos da classe

def instance01(problem):
    s1 = problem.initial
    for j in range(10):
        for i in range(10):
            if j == 0:
                if i == 1 or i == 9:
                    s1 = problem.result(s1, [[i, j, 'b']])
                elif i == 4:
                    s1 = problem.result(s1, [[i, j, 'c']])
                elif i == 7:
                    s1 = problem.result(s1, [[i, j, 't']])
                elif i == 8:
                    s1 = problem.result(s1, [[i, j, 'm']])
            if i == 3 and j == 9:
                s1 = problem.result(s1, [[i,j,'m']])
            if j == 4:
                if i == 6:
                    s1 = problem.result(s1, [[i, j, 't']])
                if i == 7:
                    s1 = problem.result(s1, [[i, j, 'b']])
            if j == 6:
                if i == 0:
                    s1 = problem.result(s1, [[i, j, 't']])
                if i == 2:
                    s1 = problem.result(s1, [[i, j, 'b']])
            if j == 7:
                if i == 4:
                    s1 = problem.result(s1, [[i, j, 'c']])
            if j == 8:
                if i == 7:
                    s1 = problem.result(s1, [[i, j, 't']])
            if j == 9:
                if i == 1:
                    s1 = problem.result(s1, [[i, j, 't']])
                if i == 2 or j == 3:
                    s1 = problem.result(s1, [[i, j, 'm']])
                if i == 4:
                    s1 = problem.result(s1, [[i, j, 'b']])
    return s1

if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.


    game_board = Board.parse_instance()

    problem = Bimaru(game_board)

    #goal_node = depth_first_tree_search(problem)    

    state1 = problem.result(problem.initial, [(1, 0, 'b')])
    
    #Out
    print(state1.board.adjacent_vertical_values(0,0)[0]) 

    #print(goal_node)

    # Convert array to string where each row is a line

    print(state1.board)

    pass
