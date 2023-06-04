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
        self.board.row_elements_curr = board.row_elements_curr.copy()
        self.board.col_elements_curr = board.col_elements_curr.copy()
        self.board.game_cells = np.copy(board.game_cells)   # Avoids changing previous boards
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

        self.board.one_boat = board.one_boat
        self.board.two_boat = board.two_boat
        self.board.three_boat = board.three_boat
        self.board.four_boat = board.four_boat

    def __lt__(self, other):
        return self.id < other.id


class Board:
    boats = ['T', 'B', 'L', 'M', 'R', 'C', 't', 'b', 'l', 'm', 'r', 'c']
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self):
        self.row_elements = []
        self.col_elements = []
        self.row_elements_curr = []
        self.col_elements_curr = []
        self.game_cells = np.empty((10, 10), dtype=np.chararray)

        self.one_boat = 4
        self.two_boat = 3
        self.three_boat = 2
        self.four_boat = 1
    
    def __str__(self):
        string_representation = ''#.join((str(row)) for row in self.row_elements) + '\n' + ''.join((str(col)) for col in self.col_elements) + '\n'
        string_representation += '\n'.join([''.join(row.astype(str)) for row in self.game_cells])
        return string_representation.replace('None', '_')

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.game_cells[row, col]

    def set_value(self, row: int, col: int, val: int):
        self.game_cells[row, col] = val

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
                initial_board.row_elements = list(map(int, line[1:]))
                initial_board.row_elements_curr = list(map(int, line[1:]))
            elif line[0] == 'COLUMN':
                initial_board.col_elements = list(map(int, line[1:]))
                initial_board.col_elements_curr = list(map(int, line[1:]))

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

        initial_board.count_initial_boats()
        initial_board.water_fill()
        return initial_board


    def count_initial_boats(self):
        for i in range(10):
            for u in range(10):
                val = self.get_value(i, u)
                cnt = 0

                if val == 'C': self.one_boat -= 1
                elif val == 'T':
                    while self.get_value(i+cnt, u) != None and self.get_value(i+cnt, u) != "W":
                        cnt += 1
                    if self.get_value(i + cnt - 1, u) != 'B':
                        continue
                elif val == 'L':
                    while self.get_value(i, u+cnt) != None and self.get_value(i, u+cnt) != "W":
                        cnt += 1
                    if self.get_value(i, u + cnt - 1) != 'R':
                        continue
                
                if cnt == 4:
                    self.four_boat -= 1
                elif cnt == 3:
                    self.three_boat -= 1
                elif cnt == 2:
                    self.two_boat -= 1
               
    # Used to fill with water in columns/lines where no more boats can be put
    def water_fill(self):
        
        for i in range(10):
            row_el = self.row_elements[i]
            col_el = self.col_elements[i]
            all_row = self.game_cells[i, :]
            all_col = self.game_cells[:, i]

            if row_el == 0:
                self.game_cells[i, :] = ['.' if cell is None else cell for cell in all_row]
            if col_el == 0:
                self.game_cells[:, i] = ['.' if cell is None else cell for cell in all_col]

            # for row
            filled_cells = np.count_nonzero(all_row)
            water_tips = np.count_nonzero(np.logical_or(all_row == 'W', all_row == '.'))
            if filled_cells - row_el - water_tips == 0:
                self.game_cells[i, :] = ['.' if cell is None else cell for cell in all_row]
            # for column
            filled_cells = np.count_nonzero(all_col)
            elements = self.col_elements[i]
            water_tips = np.count_nonzero(np.logical_or(all_col == 'W', all_col == '.'))
            if filled_cells - elements - water_tips == 0:
                self.game_cells[:, i] = ['.' if cell is None else cell for cell in all_col]
                    
            for u in range(10):
                if self.get_value(i, u) in self.boats:
                    self.fill_surroundings((i,u))

    def fill_surroundings(self, coord):
        x = coord[0]
        y = coord[1]
        val = self.game_cells[x, y].casefold()

        surround = self.check_surroundings(coord)

        for s in range(len(surround)):
            if surround[s] == None and val in self.boats:
                # left
                if s == 0 and val != 'r' and val != 'm': self.set_value(x, y-1, '.')  
                # right
                elif s == 1 and val != 'l' and val != 'm': self.set_value(x, y+1, '.')    
                # up
                elif s == 2 and val != 'b' and val != 'm': self.set_value(x-1, y, '.')
                # down
                elif s == 3 and val != 't' and val != 'm': self.set_value(x+1, y, '.')
                # right_up
                elif s == 4: self.set_value(x-1, y+1, '.')
                # right_down
                elif s == 5: self.set_value(x+1, y+1, '.')
                # left_up
                elif s == 6: self.set_value(x-1, y-1, '.')
                # left_down
                elif s == 7: self.set_value(x+1, y-1, '.')

    # Devolve duas listas, uma de objetos adjacentes presentes numa lista dada
    # outra de todos os espacos adjacentes das coordenadas
    def check_surroundings(self, coord):
        x = coord[0]
        y = coord[1]
        # Surround sera [left, right, up, down, r_up, r_down, l_up, l_down]
        surround = []
        surround.extend(self.adjacent_horizontal_values(x,y))
        surround.extend(self.adjacent_vertical_values(x,y))
        surround.extend(self.adjacent_diagonal_values(x,y))
        return surround

    def check_lateral(self, origin, size: int, direction: str):

        if direction == "col":
            return self.col_elements_curr[origin[1]] >= size
        else:
            return self.row_elements_curr[origin[0]] >= size

    def check_possible_position(self, row: int, col: int):
        return self.game_cells[row][col] != '.' and self.game_cells[row][col] != 'W' and\
             (self.game_cells[row][col] == None or self.game_cells[row][col].isupper())

    def check_limits_horizontal(self, row: int, col: int, size: int):
        a = True
        b = True
        if col < 10 - size:
            a = self.game_cells[row][col + size] == None \
                or self.game_cells[row][col + size] == '.' or self.game_cells[row][col + size] == 'W'\
                or self.game_cells[row][col + size] == 'Out'
        if col != 0:
            b = self.game_cells[row][col - 1] == None \
                or self.game_cells[row][col - 1] == '.' or self.game_cells[row][col - 1] == 'W'\
                or self.game_cells[row][col - 1] == 'Out'
        return a and b

    def check_limits_vertical(self, row: int, col: int, size: int):
        a = True
        b = True
        if row < 10 - size:
            a = self.game_cells[row + size][col] == None \
                or self.game_cells[row + size][col] == '.' or self.game_cells[row + size][col] == 'W'\
                or self.game_cells[row + size][col] == 'Out'
        if row != 0:
            b = self.game_cells[row - 1][col] == None \
                or self.game_cells[row - 1][col] == '.' or self.game_cells[row - 1][col] == 'W'\
                or self.game_cells[row - 1][col] == 'Out'
        return a and b

    def find_boat_vertical(self, col: int, size: int):
        actions = []
        for i in range(11-size):
            if not self.check_limits_vertical(i, col, size):
                continue
            if self.check_possible_position(i, col):
                act = []
                for u in range(size):
                    # se a intersecao de check_surroundings com boats
                    if self.check_possible_position(i + u, col):
                        val = self.game_cells[i + u, col]
                        if val == None:
                            if u == 0:
                                # se qq posicao sem ser down estiver em boats, break
                                val = 't'
                            elif u == size - 1:
                                val = 'b'
                            else:
                                val = 'm'
                        else:
                            if u == 0 and val != 'T':
                                act = None
                                break
                            elif u == size - 1 and val != 'B':
                                act = None
                                break
                            elif  u != 0 and  u != size - 1 and val != 'M':
                                act = None
                                break
                        act.append([i + u, col, val])
                    else:
                        act = None
                        break
                if act != None:
                    all_upper = True
                    for a in act:
                        if a[2].islower():
                            all_upper = False
                    if not all_upper: 
                        actions.append(act)

        return actions
        
    def find_boat_horizontal(self, row: int, size: int):
        actions = []
        for i in range(11-size):
            if not self.check_limits_horizontal(row, i, size):
                continue
            if self.check_possible_position(row, i):
                act = []
                for u in range(size):
                    if self.check_possible_position(row, i + u):
                        val = self.game_cells[row][i + u]
                        if val == None:
                            if u == 0:
                                val = 'l'
                            elif u == size - 1:
                                val = 'r'
                            else:
                                val = 'm'
                        else:
                            if u == 0 and val != 'L':
                                act = None
                                break
                            elif u == size - 1 and val != 'R':
                                act = None
                                break
                            elif  u != 0 and  u != size - 1 and val != 'M':
                                act = None
                                break
                        
                        act.append([row, i + u, val])
                    else:
                        act = None
                        break
                if act != None:
                    all_upper = True
                    for a in act:
                        if a[2].islower():
                            all_upper = False
                    if not all_upper: 
                        actions.append(act)
        return actions

    # Encontrar n quadrados consecutivos ou usar partes oferecidas pelas HINTs
    # Recebe um BimaruState e devolve uma lista com as possiveis posicoes de {n}Boat
    def find_boat_actions(self, size: int):
        actions = []

        for i in range(10):
            if self.check_lateral((0, i), size, "col"):
                actions.extend(self.find_boat_vertical(i, size))
            if self.check_lateral((i, 0), size, "row"):
                actions.extend(self.find_boat_horizontal(i, size))

        return actions
    

class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        # Actions para 4_boat, depois result, depois actions para 3_boat, repeat...
        if state.board.four_boat != 0:
            return state.board.find_boat_actions(4)
        
        # Find 3 consecutive squares in a row or col or use boat parts given by hints
        if state.board.three_boat != 0:
            return state.board.find_boat_actions(3)

        # Find 2 consecutive squares in a row or col or use boat parts given by hints
        if state.board.two_boat != 0:
            return state.board.find_boat_actions(2)
        
        if state.board.one_boat != 0:
            actions = []
            for i in range(10):
                for u in range(10):
                    if state.board.get_value(i, u) == None:
                        actions.append([[i, u, 'c']])
            
            return actions

        return []

    # action: list of list of int/char
    # Place a boat, surrounded by water
    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        new_state = BimaruState(state.board)
        board = new_state.board
        for a in action:
            board.set_value(a[0], a[1], a[2])
            board.row_elements_curr[a[0]] -= 1
            board.col_elements_curr[a[1]] -= 1

        board.water_fill()
        #print(board)

        # When four_boat is placed, state.board.four_boat -= 1
        size = len(action)
        if size == 4: board.four_boat -= 1
        elif size == 3: board.three_boat -= 1
        elif size == 2: board.two_boat -= 1
        elif size == 1: board.one_boat -= 1
        
        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        one_boat = state.board.one_boat
        two_boat = state.board.two_boat
        three_boat = state.board.three_boat
        four_boat = state.board.four_boat

        if (one_boat+two_boat+three_boat+four_boat) != 0:
            return False

        # Verificar que todas as colunas e linhas tem o maximo possivel de barcos
        # Verificar que apenas C/c's estao rodeados de '.' ou 'W'
        for i in range(10):
            row_elements = state.board.row_elements[i]
            col_elements = state.board.col_elements[i]
            for u in range(10):
                val = state.board.get_value(i, u)
                col_val = state.board.get_value(u, i)
                if val == None or col_val == None:
                    return False

                if (val.isupper() or val.islower()) and val != 'W':
                    row_elements -= 1
                if (col_val.isupper() or col_val.islower()) and col_val != 'W':
                    col_elements -= 1

                valid = False
                if val != 'W' and val != '.':
                    if val.casefold() != 'c':
                        for s in state.board.check_surroundings((i,u)):
                            if s != '.' and s != 'W' and s != None:
                                valid = True
                                break
                        if not valid:
                            return valid
                    elif val.casefold() == 'c':
                        for s in state.board.check_surroundings((i,u)):
                            if s != '.' and s != 'W' and s != None and s != "Out":
                                return False
            if row_elements != 0 or col_elements != 0:
                return False

        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass 

if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    game_board = Board.parse_instance()

    #print(game_board)

    problem = Bimaru(game_board)

    goal_node = depth_first_tree_search(problem)    

    #goal_node = breadth_first_tree_search(problem)  
   
    #print(goal_node.path_cost)
    print(goal_node.state.board)

    # Convert array to string where each row is a line

    # diff ../../instances-students/instance01.out results/01.txt 

    pass
