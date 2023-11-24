from enum import Enum
from typing import Optional

class FieldColor(Enum):
    BLACK = "Black"
    WHITE = "White"

field_black=FieldColor.BLACK
field_white=FieldColor.WHITE

class CheckerColor(Enum):
    BLACK="Black"
    WHITE="White"

checker_black=CheckerColor.BLACK
checker_white=CheckerColor.WHITE

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
    # def add_stack(self,stack)->O
    def __str__(self):
        #idx=len(self.stack)
        #if idx>0:
        #   return f"{self.stack[idx-1]}"
        return f"{self.field_type.value}"


class Board():
    def __init__(self,num_of_fields):
        self.num_of_fields=num_of_fields
        self.fields=[[Field(field_black,"black" if row%2==0 else "white") if (row+col)%2==1 else Field(field_white," ") \
                                                                        for col in range (num_of_fields)]       \
                                                                        for row in range (num_of_fields)]
    def __str__(self):
        board_str = ""
        for row in self.fields:
            board_str += ' '.join(str(field) for field in row) + "\n"
        return board_str

class Player():
    def __init__(self,checker_color):
        self.checker_color=checker_color
    def __str__(self):
        return f"{self.checker_color.value}"

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
        print("1.White")
        print("2.Black")
        return int(input())

    def __str__(self):
        return f"Player 1: {self.player1.checker_color.name}\n" \
               f"Player 2: {self.player2.checker_color.name}\n\n" \
               f"{self.board}"   
def main():
    game = Game()
    game.start()
    
if __name__=="__main__":
    main()
