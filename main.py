import copy
from enum import Enum
from queue import Queue
from typing import Optional
import json
from collections import deque

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
        if length ==0:
            self.stack=checker
        elif length+len(checker)==8:
            self.stack=[]
            return True,checker
        else:
            self.stack+=checker
        return False,None

    def is_empty(self)->bool:
        if len(self.stack)>0:
            return False
        return True

    def __str__(self):
        character='.' if self.field_type==field_black else ' '
        matrix = [[character for _ in range(3)] for _ in range(3)]
        for i, checker in enumerate(self.stack):
            if i < 8:  
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
    def __init__(self,board,current_player,player1,player2):
        self.board=board
        self.current_player=current_player
        self.player1=player1
        self.player2=player2
        self.winner=None

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
    
    def bounds_check_and_get_field(self, row, col):
        length=self.board.num_of_fields
        if 0 <= row < length and 0 <= col < length:
            return self.board.fields[row][col]
        else:
            return None 
    
    def bfs(self, start_row, start_col, directions):
        queue = Queue()
        visited = set()
        paths = []

        queue.put([(start_row, start_col, 0)])

        while not queue.empty():
            path = queue.get()
            curr_row, curr_col, distance = path[-1]

            if not self.board.fields[curr_row][curr_col].is_empty() and distance > 0:
                paths.append(path)
                continue 

            if (curr_row, curr_col) not in visited:
                visited.add((curr_row, curr_col))

                for _, (dr, dc) in directions.items():
                    new_row, new_col = curr_row + dr, curr_col + dc

                    if 0 <= new_row < self.board.num_of_fields and 0 <= new_col < self.board.num_of_fields:
                        queue.put(path + [(new_row, new_col, distance + 1)])

        valid_paths = [path for path in paths if not self.board.fields[path[-1][0]][path[-1][1]].is_empty()]

        return valid_paths 

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

        if start_field.stack[stack_pos] != self.current_player.checker_color:
            return False, "You do not own the checker you want to move."

        dir_offsets = {"GL": (-1, -1), "GD": (-1, 1), "DL": (1, -1), "DD": (1, 1)}
        if direction not in dir_offsets:
            return False, "Invalid direction."

        delta_row, delta_col = dir_offsets[direction]
        target_row_index = row_index + delta_row
        target_col_index = col_index + delta_col
        target_field = self.bounds_check_and_get_field(target_row_index,target_col_index)

        if not target_field:
            return False, "Out of bounds"

        if not (0 <= target_row_index < self.board.num_of_fields and 0 <= target_col_index < self.board.num_of_fields):
            return False, "Target position is outside the board boundaries."

        if (target_row_index + target_col_index) % 2 != 0:
            return False, "Can only move to dark squares."

        if len(target_field.stack) > 0:  
            if len(start_field.stack) - stack_pos + len(target_field.stack) >= 9:
                return False, "Cannot form a stack of nine or more."
            
        if not target_field.is_empty():
            if len(target_field.stack) > 0:
                if (
                    len(start_field.stack) > stack_pos
                ):
                    if stack_pos > 0 and target_field.stack[-1] >= start_field.stack[stack_pos]:
                        return False, "Invalid move: The moving checker must be on top of the stack."
                    num_checkers_to_move = len(start_field.stack[stack_pos:])

                    resulting_stack_size = len(target_field.stack) + num_checkers_to_move

                    if resulting_stack_size <= 8:
                        return True, "Valid move."
                    else:
                        return False, "Cannot move the stack as it exceeds the maximum size."
                else:
                    return False, "Index out of bounds."
        else:
            possible_moves = {"GL": (-1, -1), "GD": (-1, 1), "DL": (1, -1), "DD": (1, 1)}

            temp=possible_moves.pop(direction)
                
            for _, (move_row, move_col) in possible_moves.items():
                new_row, new_col = row_index + move_row, col_index + move_col

                if 0 <= new_row < self.board.num_of_fields and 0 <= new_col < self.board.num_of_fields:
                    move_field = self.board.fields[new_row][new_col]
                    if not move_field.is_empty():
                        return False, "There is a non-empty, adjacent field."
                else:
                    return False, "Move is outside the board boundaries."
                
            possible_moves[direction]=temp

            all_paths=self.bfs(row_index,col_index,possible_moves)

            shortest_paths=sorted(all_paths,key=lambda x:len(x))
            
            shortest_path=shortest_paths[0]

            paths = [path for path in shortest_paths if len(path) == len(shortest_path)]

            move_direction=possible_moves[direction]
            
            for path in paths:
                new_row,new_col,_=path[1]
                test_row=row_index+move_direction[0]
                test_col=col_index+move_direction[1]
                if (test_row,test_col)==(new_row,new_col):
                    return True, "Valid move."

            return False, "There is no valid move."

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
    
    def add_point(self,last_checker):
        if self.player0.checker==last_checker:
            self.player0.score+=1
            return
        self.player1.score+=1 


    def move(self, row, col, stack_pos, direction):
        row_index = ord(row) - ord('A')
        col_index = col - 1
        start_field = self.board.fields[row_index][col_index]

        stack_to_move = start_field.stack[stack_pos:]
        start_field.stack = start_field.stack[:stack_pos]

        target_row_index, target_col_index = self.calculate_target_position(row_index, col_index, direction)

        target_field = self.board.fields[target_row_index][target_col_index]

        exceeded,last_checker=target_field.add_checker(stack_to_move)
        if exceeded:
            return True, last_checker
        return False,None

    def execute_move(self,move):
        row, col, stack_pos, direction = move
        valid_move,message= self.is_valid_move(row, col, stack_pos, direction)
        if valid_move:
            exceeded, last_checker=self.move(row, col, stack_pos, direction)
            if exceeded:
                self.add_point(last_checker)
        else:
            print(message)
        return valid_move
    
    def to_json(self):
        return {
            "board_size": self.board.num_of_fields,
            "current_player": self.current_player.checker_color.value,
            "player1_score": self.player1.score,
            "player2_score": self.player2.score,
            "board_fields": [
                {"type": field.field_type.value, "stack": [checker.value for checker in field.stack]}
                for row in self.board.fields
                for field in row
            ]
        }

    def from_json(self, data):
        board_size = data["board_size"]
        current_player_color = data["current_player"]
        player1_score = data["player1_score"]
        player2_score = data["player2_score"]
        board_fields = data["board_fields"]

        player1 = Player(CheckerColor.X)
        player1.score=player1_score
        player2 = Player(CheckerColor.O)
        player2.score=player2_score

        current_player = player1 if current_player_color == CheckerColor.X.value else player2

        board = Board(board_size)

        fields = []
        for field_data in board_fields:
            field_type = FieldColor(field_data["type"])
            stack = [CheckerColor(checker) for checker in field_data["stack"]]
            fields.append(Field(field_type, stack))

        board.fields = [fields[i:i + board_size] for i in range(0, len(fields), board_size)]

        return Game(board, current_player, player1, player2)

    def save_to_json(self, filename):
        with open(filename, "w") as json_file:
            json.dump(self.to_json(), json_file, indent=2)

    def load_from_json(self, filename):
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        return self.from_json(data)
    
    def generate_moves_from_field(self, row, col):
        moves_from_field = []

        field = self.bounds_check_and_get_field(row, col)
        if not field or field.is_empty():
            return moves_from_field

        for stack_pos in range(len(field.stack)):
            if field.stack[stack_pos] == self.current_player.checker_color:
                for direction in ["GL", "GD", "DL", "DD"]:
                    valid_move, _ = self.is_valid_move(chr(65 + row), col + 1, stack_pos, direction)
                    if valid_move:
                        moves_from_field.append((chr(65 + row), col + 1, stack_pos, direction))

        return moves_from_field

    def generate_all_moves(self):
        all_moves = []

        for row in range(self.board.num_of_fields):
            for col in range(self.board.num_of_fields):
                if self.board.fields[row][col].field_type==field_black:
                    field=self.bounds_check_and_get_field(row,col)
                    if not field:
                       continue 
                    if not field.is_empty() and self.current_player.checker_color in field.stack:
                        moves_from_field = self.generate_moves_from_field(row, col)
                        all_moves.extend(moves_from_field)
        return all_moves
    
    def copy_game(game):
        new_game = Game()
        new_game.board = copy.deepcopy(game.board)
        new_game.current_player = copy.deepcopy(game.current_player)
        new_game.player1 = copy.deepcopy(game.player1)
        new_game.player2= copy.deepcopy(game.player2)
        new_game.winner = game.winner
        return new_game
    
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
        game.execute_move(move)
        game.switch_player()
    print(game)
    if game.won():
        print(f"{game.winner} has won!")
    
if __name__=="__main__":
    main()