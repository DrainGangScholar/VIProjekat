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
        length=len(self.stack)
        if length+1<8:
            self.stack.append(checker)
            return None
        else:
            self.stack=[]
            return checker

    def is_empty(self)->bool:
        if len(self.stack)>0:
            return False
        return True

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

    def empty(self):
        for row in self.fields:
            for field in row:
                if not field.is_empty():
                    return False
        return True

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

    def input_move(self):
        row = input("Enter row (A, B, C, ...): ").upper()
        col = int(input("Enter column (1, 2, 3, ...): "))
        stack_pos = int(input("Enter stack position (0, 1, 2, ...): "))
        direction = input("Enter direction (GL, GD, DL, DD): ").upper()
        return row, col, stack_pos, direction
    
    def is_valid_move(self, start_row, start_col, stack_pos, direction):
        row_index = ord(start_row) - ord('A')
        col_index = start_col - 1
        start_field = self.board.fields[row_index][col_index]

        if not (0 <= row_index < self.board.num_of_fields and 0 <= col_index < self.board.num_of_fields):
            return False, "Move is outside the board boundaries."

        if (row_index + col_index) % 2 != 0:
            return False, "Can only move on dark squares."

        start_field = self.board.fields[row_index][col_index]
        if len(start_field.stack) == 0:
            return False, "No stack to move."

        if start_field.stack[0] != self.current_player.checker_color:
            return False, "You do not own the bottom checker in the stack."

        dir_offsets = {"GL": (-1, -1), "GD": (-1, 1), "DL": (1, -1), "DD": (1, 1)}
        if direction not in dir_offsets:
            return False, "Invalid direction."

        delta_row, delta_col = dir_offsets[direction]
        target_row_index = row_index + delta_row
        target_col_index = col_index + delta_col

        if not (0 <= target_row_index < self.board.num_of_fields and 0 <= target_col_index < self.board.num_of_fields):
            return False, "Target position is outside the board boundaries."

        if (target_row_index + target_col_index) % 2 != 0:
            return False, "Can only move to dark squares."

        target_field = self.board.fields[target_row_index][target_col_index]

        if len(target_field.stack) > 0:  
            if len(start_field.stack) - stack_pos + len(target_field.stack) >= 9:
                return False, "Cannot form a stack of nine or more."
            
        if start_field.stack[stack_pos] != self.current_player.checker_color:
            return False, "It's not your turn."

        return True, "Valid move."
    
    def won(self):
        num_of_checkers=((self.board_size-2)*self.board_size/2)
        max_score=num_of_checkers/8
        win_score=(2*max_score)//3

        if(self.player1.score>win_score):
            self.winner=self.player1
            return True

        elif(self.player2.score>win_score):
            self.winner=self.player2
            return True

        return False
    
    def is_over(self):
        if self.board.empty():
            return False
        if not self.won():
            return False
        return True

    def switch_player(self):
        self.current_player=self.player1 if self.current_player == self.player2 else self.player2

    def get_move(self):
        print(f"{self.current_player}'s turn:")
        return self.input_move()
    
    def calculate_target_position(self, row_index, col_index, direction):
        dir_offsets = {"GL": (-1, -1), "GD": (-1, 1), "DL": (1, -1), "DD": (1, 1)}
        delta_row, delta_col = dir_offsets[direction]
        return row_index + delta_row, col_index + delta_col

    def move(self, row, col, stack_pos, direction):
        row_index = ord(row) - ord('A')
        col_index = col - 1
        start_field = self.board.fields[row_index][col_index]

        checker = start_field.stack.pop(stack_pos)
        target_row_index, target_col_index = self.calculate_target_position(row_index, col_index, direction)

        target_field = self.board.fields[target_row_index][target_col_index]
        target_field.add_checker(checker)

        #self.current_player.add_score(len(target_field.stack))

    def execute_move(self,move):
        row, col, stack_pos, direction = move
        valid_move,message= self.is_valid_move(row, col, stack_pos, direction)
        if valid_move:
            self.move(row, col, stack_pos, direction)
        else:
            print(message)
        return valid_move

    def __str__(self):
        return f"{self.board}\n" \
               f"Player 1: {self.player1}\n" \
               f"Player 2: {self.player2}\n\n" 

def main():
    game = Game()
    game.start()
    while not game.is_over():
        print(game)
        move=game.get_move()
        valid=game.execute_move(move)
        while not valid:
            move=game.get_move()
            valid=game.execute_move(move)
        game.switch_player()
        # valid_move, error_message = execute_move(game, move)

        # while not valid_move:
        #     print(error_message)
        #     move = current_player.get_move()
        #     #valid_move, error_message = execute_move(game, move)
    print(game)
    if game.won():
        print(f"{game.winner} has won!")
    
if __name__=="__main__":
    main()