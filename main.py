from enum import Enum
from typing import Optional

class FieldColor(Enum):
    BLACK = "Black"
    WHITE = "White"

field_black=FieldColor.BLACK
field_white=FieldColor.WHITE

class CheckerColor(Enum):
    X="X"
    O="O"

checker_black=CheckerColor.X
checker_white=CheckerColor.O

class Field():
    def __init__(self,field_type,checker):
        self.field_type=field_type
        self.stack=[]
        if checker is not None:
            self.stack.append(checker)

    def add_checker(self,checker)->Optional[CheckerColor]:
        length=len(stack)
        if length+1<8:
            self.stack.append(checker)
            return None
        else:
            self.stack=[]
            return checker

    def __str__(self):
        character='.' if self.field_type==field_black else ' '
        matrix = [[character for _ in range(3)] for _ in range(3)]
        for i, checker in enumerate(self.stack):
            if i < 9:  
                row, col = divmod(i, 3)
                matrix[row][col] = checker.value
        return "\n".join(''.join(row) for row in matrix) + "\n"


class Board():
    def __init__(self, num_of_fields):
        self.num_of_fields = num_of_fields
        self.fields = []

        for row in range(num_of_fields):
            row_fields = []
            for col in range(num_of_fields):
                is_black_field = (row + col) % 2 == 0
                field = Field(field_black if is_black_field else field_white, None)

                if is_black_field and 0 < row < num_of_fields - 1:
                    checker = checker_black if row % 2 == 1 else checker_white
                    field.stack.append(checker)

                row_fields.append(field)
            self.fields.append(row_fields)

    def str(self):
        board_str = ""
        for row in self.fields:
            for _ in range(3):
                board_str += ' '.join(field.__str__().split('\n')[_] for field in row) + "\n"
        return board_str


    def __str__(self):
        column_labels = "  " + "   ".join(str(i + 1) for i in range(self.num_of_fields))
        board_str = column_labels + "\n"

        for i, row in enumerate(self.fields):
            row_label = chr(65 + i)  
            row_str_lines = ['' for _ in range(3)] 
            for field in row:
                field_lines = field.__str__().split('\n')
                for j in range(3):
                    row_str_lines[j] += field_lines[j] + " "

            board_str += f"{row_label} " + "\n  ".join(row_str_lines) + "\n"

        return board_str

class Player():
    def __init__(self,checker_color):
        self.checker_color=checker_color
        self.score=0

    def add_score(self, points):
        self.score += points

    def __str__(self):
        return f"{self.checker_color.value} - Score: {self.score}"

class Game():
    def __init__(self):
        self.board=None
        self.winner=None

    def start(self):
        self.get_board_size()
        player1=Player(checker_white)
        player2=Player(checker_black)
        if(1==self.get_first_player()):
            self.init(player1,player2)
        else:
            self.init(player2,player1)
        print(self)

    def init(self,player1,player2):
        self.board=Board(self.board_size)
        self.current_player=player1
        self.player1=player1
        self.player2=player2

    def get_board_size(self):
        self.board_size = int(input("Board size: "))

    def get_first_player(self):
        print("Choose who goes first: ")
        print("1.O")
        print("2.X")
        return int(input())

#   def make_move(self,start_row,start_col,

    def __str__(self):
        return f"{self.board}\n" \
               f"Player 1: {self.player1}\n" \
               f"Player 2: {self.player2}\n\n" 

def main():
    game = Game()
    game.start()

if __name__=="__main__":
    main()
