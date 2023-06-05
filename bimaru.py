# 4Boat / four_boat => boat of size 4
# 3Boat / three_boat => boat of size 3
# 2Boat / two_boat => boat of size 2
# 1Boat / one_boat => boat of size  1

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

        self.is_row_completed = board.is_row_completed
        self.is_col_completed = board.is_col_completed
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
        self.is_row_completed = [False] * 10
        self.is_col_completed = [False] * 10

        self.one_boat = 4
        self.two_boat = 3
        self.three_boat = 2
        self.four_boat = 1
    
    def __str__(self):
        string_representation = ''#.join((str(row)) for row in self.row_elements) + '\n' + ''.join((str(col)) for col in self.col_elements) + '\n'
        string_representation += '\n'.join([''.join(row.astype(str)) for row in self.game_cells])
        return string_representation.replace('None', '_')

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posicao do tabuleiro."""

        return self.game_cells[row, col]

    def set_value(self, row: int, col: int, val: int):
        self.game_cells[row, col] = val

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo, respectivamente."""

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
        """Devolve os valores imediatamente a esquerda e a direita, respectivamente."""

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
        """Devolve os valores das diagonais adjacentes da celula"""

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


    # Count how many boats are initially given and decreases the correspondent counter
    def count_initial_boats(self):
        for i in range(10):
            for u in range(10):
                val = self.get_value(i, u)
                cnt = 0

                if val == 'C': self.one_boat -= 1
                elif val == 'T':
                    s = self.check_surroundings((i, u))[3]
                    while s != "Out" and \
                    self.get_value(i+cnt, u) != None and self.get_value(i+cnt, u) != "W":
                        cnt += 1

                    if not self.check_limits_vertical(i, u, cnt):
                        continue

                elif val == 'L':
                    s = self.check_surroundings((i, u))[1]
                    while s != "Out" and\
                    self.get_value(i, u+cnt) != None and self.get_value(i, u+cnt) != "W":
                        cnt += 1
                    if not self.check_limits_horizontal(i, u, cnt):
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
            if self.is_row_completed[i] or self.is_col_completed[i]:
                continue

            row_el = self.row_elements[i]
            col_el = self.col_elements[i]
            all_row = self.game_cells[i, :]
            all_col = self.game_cells[:, i]

            if row_el == 0:
                self.game_cells[i, :] = ['.' if cell is None else cell for cell in all_row]
                self.is_row_completed[i] = True
            if col_el == 0:
                self.game_cells[:, i] = ['.' if cell is None else cell for cell in all_col]
                self.is_col_completed[i] = True

            # for row
            filled_cells = np.count_nonzero(all_row)
            if filled_cells == 10:
                self.is_row_completed[i] = True
            else:   
                water_tips = np.count_nonzero(np.logical_or(all_row == 'W', all_row == '.'))
                if filled_cells - row_el - water_tips == 0:
                    self.game_cells[i, :] = ['.' if cell is None else cell for cell in all_row]
                    self.is_row_completed[i] = True

            # for column
            filled_cells = np.count_nonzero(all_col)
            if filled_cells == 10:
                self.is_col_completed[i] = True
            else:
                elements = self.col_elements[i]
                water_tips = np.count_nonzero(np.logical_or(all_col == 'W', all_col == '.'))
                if filled_cells - elements - water_tips == 0:
                    self.game_cells[:, i] = ['.' if cell is None else cell for cell in all_col]
                    self.is_col_completed[i] = True
                    
            for u in range(10):
                if None not in self.check_surroundings((i,u)):
                   continue
                if self.get_value(i, u) in self.boats:
                    self.fill_surroundings((i,u))
            
    # Surrounds a given square with water. The adjacent squares 
    # filled depend on the given square itself
    def fill_surroundings(self, coord):
        x = coord[0]
        y = coord[1]
        val = self.game_cells[x, y].casefold()

        surround = self.check_surroundings(coord)
        for s in range(8):
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

    # Returns a list with all adjacent squares
    def check_surroundings(self, coord):
        x = coord[0]
        y = coord[1]
        # Surround is [left, right, up, down, r_up, r_down, l_up, l_down]
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

    def check_possible_position(self, val: str):
        return not (val == 'W' or val == '.' or (val != None and val.islower()))

    def check_limits_horizontal(self, row: int, col: int, size: int):
        a = True
        b = True
        # Check in front
        if col < 10 - size:
            front = self.game_cells[row][col + size]
            a = front == None or front == '.' or front == 'W'
        # Check behind
        if col != 0:
            back = self.game_cells[row][col - 1]
            b = back == None or back == '.' or back == 'W'
        return a and b

    def check_limits_vertical(self, row: int, col: int, size: int):
        a = True
        b = True
        # Check in front
        if row < 10 - size:
            front = self.game_cells[row + size][col]
            a = front == None or front == '.' or front == 'W' 
        # Check behind
        if row != 0:
            back = self.game_cells[row - 1][col]
            b = back == None or back == '.' or back == 'W'
        return a and b

    def check_general(self, state: BimaruState):
        # Verify if every column and row is at max capacity of boat parts
        # Verify that only C/c's are totally surrounded by water
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
                    elif val.casefold() == 'm' or val == 'b' and ((self.game_cells[i - 1, u] == 'W'\
                    or self.game_cells[i - 1, u] == '.') and (self.game_cells[i, u - 1] == 'W'\
                    or self.game_cells[i, u - 1] == '.')): 
                        return False
                    elif val.casefold() == 'm' or val == 'r' and ((self.game_cells[i - 1, u] == 'W'\
                    or self.game_cells[i - 1, u] == '.') and (self.game_cells[i, u - 1] == 'W'\
                    or self.game_cells[i, u - 1] == '.')): 
                        return False

            if row_elements != 0 or col_elements != 0:
                return False

        return True

    # Searches each column for possible boat positions from top to bottom
    def find_boat_vertical(self, col: int, size: int):
        actions = []
        for i in range(11-size):
            surrounding = self.check_surroundings((i, col))
            val = self.get_value(i, col)
            if not self.check_possible_position(val):
                continue
            if not self.check_limits_vertical(i, col, size):
                continue
            act = []
            for u in range(size):
                val = self.get_value(i + u, col)
                if self.check_possible_position(val):
                    if val == None:
                        if u == 0:
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
        
    # Searches each row for possible boat positions from left to right
    def find_boat_horizontal(self, row: int, size: int):
        actions = []
        for i in range(11-size):
            surrounding = self.check_surroundings((row, i))
            val = self.get_value(row, i)
            
            act = []
            for u in range(size):
                val = self.game_cells[row][i + u]
                if self.check_possible_position(val):
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

    # Find n consecutive squares or use HINT given parts 
    # Receives a BimaruState and returns a list with the possible {n}Boat positions
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
        #if not state.board.check_general(state):
            #return []
        
        # Actions for 4_boat, then result, then actions for 3_boat, repeat...
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
                    if state.board.one_boat == 1 and state.board.get_value(i, u) == None:
                        state.board.one_boat -= 1
                        return [[[i, u, 'c']]]
                    if state.board.get_value(i, u) == None:
                        state.board.set_value(i, u, 'c')
                        state.board.row_elements_curr[i] -= 1
                        state.board.col_elements_curr[u] -= 1
                        state.board.one_boat -= 1

            state.board.water_fill()

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

        if board.two_boat != 0:
            board.water_fill()

        size = len(action)
        # When four_boat is placed, state.board.four_boat -= 1
        if size == 4: board.four_boat -= 1
        elif size == 3: board.three_boat -= 1
        elif size == 2: board.two_boat -= 1
        #elif size == 1: board.one_boat -= 1
        
        #print()
        #print(board)
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

        # Verify if every column and row is at max capacity of boat parts
        # Verify that only C/c's are totally surrounded by water
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
                    elif val.casefold() == 'm' or val == 'b' and ((self.game_cells[i - 1, u] == 'W'\
                    or self.game_cells[i - 1, u] == '.') and (self.game_cells[i, u - 1] == 'W'\
                    or self.game_cells[i, u - 1] == '.')): 
                        return False
                    elif val.casefold() == 'm' or val == 'r' and ((self.game_cells[i - 1, u] == 'W'\
                    or self.game_cells[i - 1, u] == '.') and (self.game_cells[i, u - 1] == 'W'\
                    or self.game_cells[i, u - 1] == '.')): 
                        return False

            if row_elements != 0 or col_elements != 0:
                return False

        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        violations = 0

        print(node.state.board)

        test = np.where(node.state.board.game_cells == 'M')

        if any(value == '.' for value in node.state.board.adjacent_vertical_values(test[0], test[1])) and \
                any(value == '.' for value in node.state.board.adjacent_horizontal_values(test[0], test[1])):
            violations += 100
            print(str(violations) + ' Violation Registered!\n')
        print(node.path_cost)

        return violations

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

    pass
