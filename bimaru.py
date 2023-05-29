# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 152:
# 96196 Duarte Manuel da Cruz Costa
# 00000 Nome2

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
        self.four_boat = False

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        # TODO
        return self.game_cells[row, col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        # TODO
        up_value = self.game_cells[row - 1, col]
        down_value = self.game_cells[row + 1, col]
        return up_value, down_value

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        # TODO
        right_value = self.game_cells[row, col + 1]
        left_value = self.game_cells[row, col - 1]
        return left_value, right_value

    def adjacent_diagonal_values(self, row: int, col: int) -> (str, str, str, str):
        """Devolve os valores das diagonais adjacentes da célula"""

        # TODO WILL THIS BE NEEDED?

        r_up = self.game_cells[row - 1, col + 1]
        r_down = self.game_cells[row + 1, col - 1]
        l_up = self.game_cells[row - 1, col + 1]
        l_down = self.game_cells[row + 1, col + 1]
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
        # TODO
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

        # TODO


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        self.initial = BimaruState(board)

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO Confuso af

        possible_actions = []

        # Find indices where 'L' or 'l' occurs
        indices = np.argwhere((state.board.game_cells == 'L') | (state.board.game_cells == 'l'))

        if len(indices) != 0:
            for index in indices:
                if state.board.game_cells[index[0], index[1] + 1] is None:
                    possible_actions.append([(index[0], index[1] + 1, 'm')])
                    possible_actions.append([(index[0], index[1] + 1, 'r')])

        indices = np.argwhere((state.board.game_cells == 'R') | (state.board.game_cells == 'r'))

        if len(indices) != 0:
            for index in indices:
                if state.board.game_cells[index[0], index[1] - 1] is None:
                    possible_actions.append([(index[0], index[1] - 1, 'm')])
                    possible_actions.append([(index[0], index[1] - 1, 'l')])

        indices = np.argwhere((state.board.game_cells == 'T') | (state.board.game_cells == 't'))

        if len(indices) != 0:
            for index in indices:
                if state.board.game_cells[index[0] + 1, index[1]] is None:
                    possible_actions.append([(index[0] + 1, index[1], 'm')])
                    possible_actions.append([(index[0] + 1, index[1], 'b')])

        indices = np.argwhere((state.board.game_cells == 'B') | (state.board.game_cells == 'b'))

        if len(indices) != 0:
            for index in indices:
                if state.board.game_cells[index[0] - 1, index[1]] is None:
                    possible_actions.append([(index[0] - 1, index[1], 'm')])
                    possible_actions.append([(index[0] - 1, index[1], 't')])

        return possible_actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO

        new_state = BimaruState(state.board)

        new_state.board.game_cells[action[0][0], action[0][1]] = action[0][2]

        # TODO fill surrounding with water
        """if action[0][2] == 'b':
            new_state.board.game_cells[action[0][0] + 1, action[0][1]] = '.'
            new_state.board.game_cells[action[0][0] + 1, action[0][1] + 1] = '.'"""


        new_state.board.water_fill()

        return new_state

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO

        if state.board.game_cells.size - np.count_nonzero(state.board.game_cells) != 0:
            return False

        return True

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

    goal_node = depth_first_tree_search(problem)

    # s1 = problem.result(problem.initial, (3, 3, 'w'))

    print(goal_node)

    # Convert array to string where each row is a line
    string_representation = '\n'.join([''.join(row.astype(str)) for row in game_board.game_cells])

    print(string_representation)

    pass
