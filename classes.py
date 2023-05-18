class Board:
    """ Representação interna de uma grelha de Bimaru. """

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """ Devolve os valores imediatamente acima e abaixo,
        respectivamente. """
        # TODO
        pass

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        # TODO
        pass

    # TODO: outros metodos da classe

    def get_value(self, pos_x: int, pos_y: int):
        """ Devolve o valor na posição dos argumentos de entrada

        :return:
        """

    @staticmethod
    def parse_instance():
        """Lê a instância do problema do standard input (stdin)
        e retorna uma instância da classe Board.
        Por exemplo:
        $ python3 bimaru.py < input_T01
        > from sys import stdin
        > line = stdin.readline().split()
        """
        # TODO
        pass

